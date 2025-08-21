#pragma once
#include "hintfile-2D.cuh"

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE_X, uint32_t BLOCKSIZE_Y = BLOCKSIZE_X>
__global__ void SGEMM(const float* const __restrict A, const float* const __restrict B, float* const __restrict C, const float alpha = 1, const float beta = 0) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < K; ++i) {
            sum += A[i * M + row] * B[i + col * K];
        }
        C[col * M + row] = alpha * sum + beta * C[col * N + row];
    }
}