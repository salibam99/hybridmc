#include <cuda_runtime.h>
#include <curand.h>
#include <omp.h>
#include <iostream>

const float L = 10.0f;  // Slab length (cm)
const float sigma_t = 0.1f;  // Total cross-section (cm^-1)
const float sigma_s = 0.05f;  // Scattering
const int N_particles = 100000000;  // 10^8 for RTX 3080 Ti
const int N_bins = 100;  // Flux bins
const int block_size = 512;  // Tuned for 3080 Ti

// Kernel 1: Move particles
__global__ void move_particles(float* pos, float* dir, float* rng_buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float rnd = rng_buffer[idx];
    float dist = -logf(rnd) / (sigma_t * fabsf(dir[idx]));
    pos[idx] += dist * dir[idx];
}

// Kernel 2: Handle collisions with shadow equations
__global__ void collide_particles(float* pos, float* dir, float* flux, float* rng_buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float p = pos[idx], d = dir[idx];
    float dummy = 0.0f;  // Shadow variable
    if (p >= 0 && p <= L) {
        int bin = static_cast<int>(p / (L / N_bins));
        atomicAdd(&flux[bin], 1.0f / n);
        if (rng_buffer[idx + n] < sigma_s / sigma_t) {
            dir[idx] = rng_buffer[idx + 2 * n] * 2.0f - 1.0f;
        } else {
            pos[idx] = -1.0f;  // Absorb
            dummy = rng_buffer[idx + 2 * n] * 2.0f - 1.0f;  // Shadow: mimic scatter
        }
    } else {
        dummy = static_cast<int>(p / (L / N_bins)) * 1.0f / n;  // Shadow: mimic tally
        dummy += rng_buffer[idx + n] * rng_buffer[idx + 2 * n];  // Shadow: mimic scatter check
    }
}

int main() {
    // Unified Memory
    float *pos, *dir, *flux, *rng_buffer;
    cudaMallocManaged(&pos, N_particles * sizeof(float));
    cudaMallocManaged(&dir, N_particles * sizeof(float));
    cudaMallocManaged(&flux, N_bins * sizeof(float));
    cudaMallocManaged(&rng_buffer, 3 * N_particles * sizeof(float));
    for (int i = 0; i < N_particles; ++i) { pos[i] = 0.0f; dir[i] = 1.0f; }
    for (int i = 0; i < N_bins; ++i) flux[i] = 0.0f;

    // cuRAND setup
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, time(nullptr));

    // CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float t_rng = 0, t_move = 0, t_collide = 0, t_tally = 0;

    // RNG
    cudaEventRecord(start, stream);
    curandGenerateUniform(gen, rng_buffer, 3 * N_particles);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_rng, start, stop);

    // Event loop
    int grid_size = (N_particles + block_size - 1) / block_size;
    int active_particles = N_particles;
    int max_steps = 10;
    for (int step = 0; step < max_steps && active_particles > 0; ++step) {
        // Move
        cudaEventRecord(start, stream);
        move_particles<<<grid_size, block_size, 0, stream>>>(pos, dir, rng_buffer, N_particles);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float t_move_step;
        cudaEventElapsedTime(&t_move_step, start, stop);
        t_move += t_move_step;

        // Collide
        cudaEventRecord(start, stream);
        collide_particles<<<grid_size, block_size, 0, stream>>>(pos, dir, flux, rng_buffer, N_particles);
        cudaMemPrefetchAsync(flux, N_bins * sizeof(float), cudaCpuDeviceId, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float t_collide_step;
        cudaEventElapsedTime(&t_collide_step, start, stop);
        t_collide += t_collide_step;

        // Simplified active particle check (TODO: improve for production)
        active_particles = N_particles;
    }

    // CPU tally
    cudaEventRecord(start, stream);
    #pragma omp parallel for num_threads(16)  // i7-12K
    for (int i = 0; i < N_bins; ++i) flux[i] *= 1.0f;
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_tally, start, stop);

    // Output timings
    float t_total = t_rng + t_move + t_collide + t_tally;
    std::cout << "Total: " << t_total/1000 << "s\n";
    std::cout << "  RNG: " << t_rng/1000 << "s (" << t_rng/t_total*100 << "%)\n";
    std::cout << "  Move: " << t_move/1000 << "s (" << t_move/t_total*100 << "%)\n";
    std::cout << "  Collide: " << t_collide/1000 << "s (" << t_collide/t_total*100 << "%)\n";
    std::cout << "  Tally: " << t_tally/1000 << "s (" << t_tally/t_total*100 << "%)\n";

    // Cleanup
    cudaFree(pos); cudaFree(dir); cudaFree(flux); cudaFree(rng_buffer);
    curandDestroyGenerator(gen); cudaStreamDestroy(stream);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
