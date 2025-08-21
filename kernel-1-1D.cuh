#pragma once
#include "hintfile-1D.cuh"

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE>
__global__ void SGEMM(const float* const __restrict A, const float* const __restrict B, float* const __restrict C, const float alpha = 1, const float beta = 0) {
    const uint32_t cCol = BLOCKSIZE * blockIdx.x + (threadIdx.x / BLOCKSIZE);
	const uint32_t cRow = BLOCKSIZE * blockIdx.y + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float temp = 0.0f;
        for (uint32_t i = 0; i < K; ++i) {
            temp += A[i * M + cRow] * B[cCol * K + i];
        }
        C[cCol * M + cRow] = alpha * temp + beta * C[cCol * M + cRow];
    }
}