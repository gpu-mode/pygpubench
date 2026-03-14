// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>
#include <driver_types.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Compare expected and received for exact bitwise matches. `result` counts the number of _16-byte_ regions that contain
// mismatches.
void check_exact_match_launcher(unsigned* result, const std::byte* expected, const std::byte* received, unsigned seed, std::size_t nbytes, cudaStream_t stream);

// compare expected and received using max(atol, abs(expected)*rtol)
void check_approx_match_launcher(unsigned* result, const float* expected, const float* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream);
void check_approx_match_launcher(unsigned* result, const nv_bfloat16* expected, const nv_bfloat16* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream);
void check_approx_match_launcher(unsigned* result, const half* expected, const half* received, float r_tol, float a_tol, unsigned seed, std::size_t size, cudaStream_t stream);


void canaries(void* data, size_t size, unsigned seed, cudaStream_t stream);