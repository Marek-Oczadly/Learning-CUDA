#pragma once
#include <cuda_runtime.h>

template <unsigned int M, unsigned int N, unsigned int K>
__global__ void naiveSGEMM(float* A, float* B, float* C, float alpha = 1, float beta = 0) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}