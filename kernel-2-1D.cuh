#pragma once
#include "hintfile-1D.cuh"


template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	const uint32_t cCol = BLOCKSIZE * blockIdx.x + (threadIdx.x / BLOCKSIZE);
	const uint32_t cRow = BLOCKSIZE * blockIdx.y + (threadIdx.x % BLOCKSIZE);

	const uint32_t threadColInBlock = threadIdx.x / BLOCKSIZE;	// Does not have division in hardware so is more expensive - dont use in loops
	const uint32_t threadRowInBlock = threadIdx.x % BLOCKSIZE;	// Same for modulo. No register pressure, so it is fine to use here
	
	A += blockIdx.y * BLOCKSIZE;		// Advance A pointer to the beginning of the block row
	B += blockIdx.x * BLOCKSIZE * K;	// Advance B pointer to the beginning of the block column

	__shared__ float AS[BLOCKSIZE * BLOCKSIZE];	// Shared memory for A block
	__shared__ float BS[BLOCKSIZE * BLOCKSIZE];	// Shared memory for B block

	if constexpr (K % BLOCKSIZE == 0) {	// Simplest case where every block is a perfect fit
		float temp = 0.0f;
		for (uint32_t k = 0; k < K; k += BLOCKSIZE) {

			AS[threadIdx.x] = A[threadColInBlock * M + threadRowInBlock];
			BS[threadIdx.x] = B[threadColInBlock * K + threadRowInBlock];

			__syncthreads();		// Ensure all threads have loaded data into shared memory

			A += BLOCKSIZE * M;		// Advance A pointer to the next block row
			B += BLOCKSIZE;			// Advance B pointer to the next block column

			for (uint32_t i = 0; i < BLOCKSIZE; ++i) {
				temp += AS[i * BLOCKSIZE + threadRowInBlock] * BS[BLOCKSIZE * threadColInBlock + i];
			}

			__syncthreads();
		}
		C[cCol * M + cRow] = alpha * temp + beta * C[cCol * M + cRow];
	}
}