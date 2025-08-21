#pragma once
#include "hintfile-2D.cuh"

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE_X, uint32_t BLOCKSIZE_Y = BLOCKSIZE_X>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	const uint32_t col = blockIdx.x * BLOCKSIZE_X + threadIdx.x;
	const uint32_t row = blockIdx.y * BLOCKSIZE_Y + threadIdx.y;

	A += BLOCKSIZE_Y * blockIdx.y;
	B += BLOCKSIZE_X * blockIdx.x * K;

	__shared__ float AS[BLOCKSIZE_Y][BLOCKSIZE_X];
	__shared__ float BS[BLOCKSIZE_Y][BLOCKSIZE_X];

	if constexpr(K % BLOCKSIZE_X == 0 && M == N && M % BLOCKSIZE_Y == 0) {
		float temp = 0.0f;

		for (uint32_t k = 0; k < K; k += BLOCKSIZE_X) {
			AS[threadIdx.y][threadIdx.x] = A[threadIdx.x + M * threadIdx.y];
			BS[threadIdx.y][threadIdx.x] = B[threadIdx.x + K * threadIdx.y];

			__syncthreads();	// Ensure all threads have loaded data into shared memory

			A += BLOCKSIZE_X * M;	// Advance A pointer to the next block row
			B += BLOCKSIZE_Y;		// Move B down to the next block column

			for (uint32_t i = 0; i < BLOCKSIZE_X; ++i) {
				temp += AS[i][threadIdx.y] * BS[threadIdx.x][i];
			}
			__syncthreads();
		}
		C[col * M + row] = alpha * temp + beta * C[col * M + row];
	}
}