#pragma once
#include "hintfile-1D.cuh"


template <uint32_t M, uint32_t N, uint32_t K, 
		  uint32_t BLOCKSIZE_K, uint32_t BLOCKSIZE_M, uint32_t BLOCKSIZE_N = BLOCKSIZE_M, 
		  uint32_t TM = 8, uint32_t TN = TM, memory_location LOAD_INTO = memory_location::REGISTERS>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	
	// Evaluating constexprs. These are optimised at compile time and don't use up a register

	constexpr uint32_t tileBlockLength_M = BLOCKSIZE_M * TM;	// Tile size in the M dimension
	constexpr uint32_t tileBlockLength_N = BLOCKSIZE_N * TN;	// Tile size in the N dimension
	constexpr uint32_t tileBlockLength_K = BLOCKSIZE_K;		// Tile size in the K dimension

	constexpr uint32_t numResultsBlocktile = BLOCKSIZE_M * BLOCKSIZE_N;

	constexpr uint32_t numThreadsPerBlock_A = BLOCKSIZE_M * BLOCKSIZE_K;
	constexpr uint32_t numThreadsPerBlock_B = BLOCKSIZE_K * BLOCKSIZE_N;	
	
	if constexpr (BLOCKSIZE_M == BLOCKSIZE_N) {
		__shared__ float AS[tileBlockLength_N * tileBlockLength_K];
		__shared__ float BS[tileBlockLength_K * tileBlockLength_N];


		// Positiion of thread in each blocktile
		const uint32_t A_threadIdx_X = threadIdx.x % tileBlockLength_M;
		const uint32_t A_threadIdx_Y = threadIdx.x / tileBlockLength_M;

		const uint32_t B_threadIdx_X = threadIdx.x % tileBlockLength_K;
		const uint32_t B_threadIdx_Y = threadIdx.x / tileBlockLength_K;
	}
}