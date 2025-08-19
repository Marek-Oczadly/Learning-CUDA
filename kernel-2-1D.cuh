#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE>
__global__ void SGEMMSharedMemory(const float* A, const float* B, float* const C, const float alpha = 1.0f, const float beta = 0.0f) {
	const uint32_t cCol = BLOCKSIZE * blockIdx.x + (threadIdx.x % BLOCKSIZE); // Column index in C
	const uint32_t cRow = BLOCKSIZE * blockIdx.y + (threadIdx.x / BLOCKSIZE); // Row index in C

	const uint32_t threadRowInBlock = threadIdx.x / BLOCKSIZE;	// Does not have division in hardware so is more expensive - dont use in loops
	const uint32_t threadColInBlock = threadIdx.x % BLOCKSIZE;	// same for modulo
	
	A += blockIdx.y * K * BLOCKSIZE;	// Advance A pointer to the beginning of the block row
	B += blockIdx.x * BLOCKSIZE;		// Advance B pointer to the beginning of the block column

	__shared__ float AS[BLOCKSIZE * BLOCKSIZE];	// Shared memory for A block
	__shared__ float BS[BLOCKSIZE * BLOCKSIZE];	// Shared memory for B block

	if constexpr (K % BLOCKSIZE == 0) {	// Simplest case where every block is a perfect fit
		float temp = 0.0f;
		for (uint32_t k = 0; k < K; k += BLOCKSIZE) {

			AS[threadIdx.x] = A[threadRowInBlock * K + threadColInBlock];
			BS[threadIdx.x] = B[threadRowInBlock * N + threadColInBlock];

			__syncthreads();	// Ensure all threads have loaded data into shared memory

			A += BLOCKSIZE;	// Advance A pointer to the next block row
			B += BLOCKSIZE * N; // Advance B pointer to the next block column

			for (uint32_t i = 0; i < BLOCKSIZE; ++i) {
				temp += AS[threadRowInBlock * BLOCKSIZE + i] * BS[i * BLOCKSIZE + threadColInBlock];
			}

			__syncthreads();
		}
		C[cRow * N + cCol] = alpha * temp + beta * C[cRow * N + cCol];
	}
}