#pragma once
#include "hintfile-2D.cuh"

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE_X, uint32_t BLOCKSIZE_Y>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, const float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	
}