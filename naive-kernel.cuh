#pragma once
#include <cuda_runtime.h>
#include <cstdint>

template <uint32_t M, uint32_t N, uint32_t K>
__global__ void naiveSGEMM(const float* const A, const float* const B, float* const C, const float alpha = 1, const float beta = 0) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}