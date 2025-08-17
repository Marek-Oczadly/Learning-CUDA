#pragma once
#include <cuda_runtime.h>
#include <cstdint>

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE>
__global__ void OneDimensionalSGEMM(float* const A, float* const B, float* const C, float alpha = 1, float beta = 0) {
	const uint32_t cCol = BLOCKSIZE * blockIdx.x + (threadIdx.x % BLOCKSIZE);
	const uint32_t cRow = BLOCKSIZE * blockIdx.y + (threadIdx.x / BLOCKSIZE);

	if (cRow < M and cCol < N) {
		float temp = 0.0f;
		float 
		for (uint32_t k = 0; k < K; ++k) {
			temp += A[cRow * K + k] * B[k * N + cCol];
		}
		C[cRow * N + cCol] = alpha * temp + beta * C[cRow * N + cCol];
	}
}