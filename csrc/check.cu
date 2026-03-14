// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <vector_types.h>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cooperative_groups.h>

#include "utils.h"

__device__ std::size_t random_index(unsigned seed) {
    unsigned total_blocks = gridDim.x;
    unsigned original_block_id = blockIdx.x;
    unsigned randomized_block_id = (original_block_id * 2654435761u + seed) % total_blocks;
    return randomized_block_id * blockDim.x + threadIdx.x;
}

// bitwise exact match
__global__ void check_exact_match_kernel(unsigned* result, const uint4* expected, const uint4* received, unsigned seed, std::size_t size) {
    std::size_t idx = random_index(seed);
    if (idx >= size) return;
    uint4 a = expected[idx];
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> res(*result);

#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif

    uint4 b = received[idx];
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
        ++res;
    }
}

template<typename Float>
__global__ void check_approx_match_kernel(unsigned* result, const Float* expected, const Float* received, float r_tol, float a_tol, unsigned seed, std::size_t size) {
    std::size_t idx = random_index(seed);
    if (idx >= size) return;
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> res(*result);
    float a = static_cast<float>(expected[idx]);

    // Nan is expected is wildcard for arbitrary results
    if (isnan(a))
        return;

#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif

    float b = static_cast<float>(received[idx]);
    if (!isfinite(a)) {
        if (a != b) {
            ++res;
            return;
        }
    }
    if (!isfinite(b)) {
        ++res;
        return;
    }

    // both are zero
    if (a == 0 && b == 0) return;

    // ok, a and b are finite, and at least one is not zero
    float diff = fabsf(a - b);
    if (diff > r_tol * fabsf(a) && diff > a_tol) {
        ++res;
    }
}


void check_exact_match_launcher(unsigned* result, const std::byte* expected, const std::byte* received, unsigned seed, std::size_t nbytes, cudaStream_t stream) {
    if (nbytes % sizeof(uint4) != 0) {
        throw std::runtime_error("Expected number of bytes to be divisible by 16");
    }
    if (reinterpret_cast<std::uintptr_t>(expected) % 16 != 0) {
        throw std::runtime_error("Expected pointer must be aligned to 16");
    }
    if (reinterpret_cast<std::uintptr_t>(received) % 16 != 0) {
        throw std::runtime_error("Received pointer must be aligned to 16");
    }

    int threads = 256;
    int blocks = cuda::ceil_div(nbytes / 16, threads);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.blockDim = dim3(threads, 1, 1);
    config.gridDim = dim3(blocks, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    CUDA_CHECK(cudaLaunchKernelEx(&config, check_exact_match_kernel, result, reinterpret_cast<const uint4*>(expected), reinterpret_cast<const uint4*>(received), seed, nbytes / sizeof(uint4)));
}

template<typename Float>
void check_approx_match_launcher_tpl(unsigned* result, const Float* expected, const Float* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream) {
    if ( !(a_tol >= 0) ) throw std::runtime_error("Absolute tolerance must be non-negative");
    if ( !(r_tol >= 0) ) throw std::runtime_error("Relative tolerance must be non-negative");
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.blockDim = dim3(threads, 1, 1);
    config.gridDim = dim3(blocks, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    CUDA_CHECK(cudaLaunchKernelEx(&config, check_approx_match_kernel<Float>, result, expected, received, r_tol, a_tol, seed, size));
}

void check_approx_match_launcher(unsigned* result, const float* expected, const float* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream) {
    check_approx_match_launcher_tpl<float>(result, expected, received, r_tol, a_tol, seed, size, stream);
}

void check_approx_match_launcher(unsigned* result, const nv_bfloat16* expected, const nv_bfloat16* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream) {
    check_approx_match_launcher_tpl<nv_bfloat16>(result, expected, received, r_tol, a_tol, seed, size, stream);
}

void check_approx_match_launcher(unsigned* result, const half* expected, const half* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream) {
    check_approx_match_launcher_tpl<half>(result, expected, received, r_tol, a_tol, seed, size, stream);
}

/// pseudo-random invertible transformation of a very small subset of input cache lines.
__global__ void canaries_kernel(uint4* data, size_t size, unsigned seed) {
    auto grid = cooperative_groups::this_grid();
    constexpr unsigned CACHE_LINE_SIZE = 128;
    constexpr unsigned STRIDE = 256;
    constexpr unsigned THREADS_PER_LINE = 8;
    constexpr unsigned LINES_PER_BLOCK = 256 / THREADS_PER_LINE;

    grid.sync();
    // using 16-byte loads/stores, each group of 8 threads handles one cache line
    const unsigned tg = threadIdx.x / THREADS_PER_LINE;
    const unsigned rtg = tg ^ (seed & 0x1f); // "random" index  [0, 31]
    const unsigned lb = seed >> 5u;

    unsigned num_blocks  = cuda::ceil_div(size, LINES_PER_BLOCK * CACHE_LINE_SIZE * STRIDE);
    for (unsigned bidx = blockIdx.x; bidx < num_blocks; bidx += gridDim.x) {
        unsigned idx = bidx * LINES_PER_BLOCK + tg;
        // we pick 1 out of every 256 cache lines
        unsigned cache_line = idx * STRIDE + (rtg * 48 + lb) % STRIDE;
        unsigned addr = cache_line * CACHE_LINE_SIZE / sizeof(uint4) + threadIdx.x % 8;
        if (addr * sizeof(uint4) < size) {
            uint4 load = data[addr];
            // self-inverse transformation
            load.x = load.x ^ seed;
            load.y = ~load.y;
            load.z = load.z ^ seed;
            load.w = ~load.w;
            data[addr] = load;
        }
    }

    grid.sync();
}

void canaries(void* data, size_t size, unsigned seed, cudaStream_t stream) {
    int block_size = 256;
    int dev, smem, max_blocks, num_sms;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));

    // allocate all available shared memory to prevent concurrent blocks
    CUDA_CHECK(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    CUDA_CHECK(cudaFuncSetAttribute(&canaries_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, &canaries_kernel, block_size, smem));
    int grid_size = max_blocks * num_sms;
    void *pArgs[] = { &data, &size, &seed};
    CUDA_CHECK(cudaLaunchCooperativeKernel(&canaries_kernel, grid_size, block_size, pArgs, smem, stream));
}