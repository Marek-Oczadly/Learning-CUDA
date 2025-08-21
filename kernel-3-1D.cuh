#pragma once
#include "hintfile-1D.cuh"


template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE, uint32_t TM = 8, bool load_into_registers = true>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	
	constexpr uint32_t tileBlockSize = BLOCKSIZE * TM;
	
	uint32_t cCol = BLOCKSIZE * blockIdx.y + (threadIdx.x / BLOCKSIZE);
	uint32_t cRow = BLOCKSIZE * blockIdx.x + (threadIdx.x % BLOCKSIZE);

	const uint32_t threadIdx_X = threadIdx.x % BLOCKSIZE;
	const uint32_t threadIdx_Y = threadIdx.x / BLOCKSIZE;

	__shared__ float AS[SQUARED(tileBlockSize)]; // Contains TM*TM elements of A for each thread
	__shared__ float BS[SQUARED(tileBlockSize)]; // Contains TM*TM elements of B for each thread

	A += tileBlockSize * blockIdx.x;
	B += tileBlockSize * K * blockIdx.y;

	// May disable if register spills occur in Nsight
	if constexpr (load_into_registers) {
		float cReg[SQUARED(TM)];
	}

	
	if constexpr (M == N && N == K && M % tileBlockSize == 0) {	// simplest case - 2 square matrices that fit all the blocks perfectly
		// Iterating the blocks across the columns of A and rows of B
		for (uint32_t k = 0; k < K; k += tileBlockSize) {
			// Loading the global memory into smem
			#pragma unroll
			for (uint32_t offset_x = 0; offset_x < tileBlockSize; offset_x+=BLOCKSIZE) {
				for (uint32_t offset_y = 0; offset_y < tileBlockSize; offset_y+=BLOCKSIZE) {
					
					uint32_t SMEM_pos = (offset_x + threadIdx_X) + (offset_y + threadIdx_Y) * tileBlockSize;
					uint32_t GMEM_pos = (offset_x + threadIdx_X) + (offset_y + threadIdx_Y) * K;	// Already confirmed that K = M = N
					// AS and BS are layed out the same way in memory as they are in global memory
					AS[SMEM_pos] = A[GMEM_pos];
					BS[SMEM_pos] = B[GMEM_pos];
				}
			}
			__syncthreads();	// Ensure all data has been loaded into shared memory

			A += tileBlockSize * M;
			B += tileBlockSize;



		}
	}
}