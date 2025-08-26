#pragma once
#include "hintfile-1D.cuh"


template <uint32_t M, uint32_t N, uint32_t K, 
		  uint32_t BLOCKSIZE, uint32_t TILESIZE_M, uint32_t TILESIZE_N = TILESIZE_M, 
		  uint32_t TM = 8, uint32_t TN = TM, memory_location LOAD_INTO = memory_location::REGISTERS>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	
	// Evaluating constexprs. These are optimised at compile time and don't use up a register
	constexpr uint32_t TILESIZE_K = BLOCKSIZE / (TILESIZE_M * TILESIZE_N);

	constexpr uint32_t BLOCKTILE_LENGTH_M = TILESIZE_M * TM;	// Tile size in the M dimension
	constexpr uint32_t BLOCKTILE_LENGTH_N = TILESIZE_N * TN;	// Tile size in the N dimension
	constexpr uint32_t BLOCKTILE_LENGTH_K = TILESIZE_K;			// Tile size in the K dimension

	constexpr uint32_t NUM_THREADS_BLOCKTILE = TILESIZE_M * TILESIZE_N;

	constexpr uint32_t BLOCKTILE_AREA_A = BLOCKTILE_LENGTH_M * BLOCKTILE_LENGTH_K;
	constexpr uint32_t BLOCKTILE_AREA_B = BLOCKTILE_LENGTH_N * BLOCKTILE_LENGTH_K;
	constexpr uint32_t BLOCKTILE_AREA_C = BLOCKTILE_LENGTH_M * BLOCKTILE_LENGTH_N;	// Size of the tile block in C
	
	if constexpr (TILESIZE_M == TILESIZE_N && M == N) {

		__shared__ float AS[BLOCKTILE_LENGTH_M * BLOCKTILE_LENGTH_K];
		__shared__ float BS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_N];


		// Position of thread in each blocktile
		const uint32_t A_threadIdx_X = threadIdx.x % BLOCKTILE_LENGTH_M;
		const uint32_t A_threadIdx_Y = threadIdx.x / BLOCKTILE_LENGTH_M;

		const uint32_t B_threadIdx_X = threadIdx.x % BLOCKTILE_LENGTH_K;
		const uint32_t B_threadIdx_Y = threadIdx.x / BLOCKTILE_LENGTH_K;

		A += blockIdx.x * BLOCKTILE_LENGTH_M;
		B += blockIdx.y * BLOCKTILE_LENGTH_N * K;

		if constexpr (K % BLOCKTILE_LENGTH_K == 0 && M % BLOCKTILE_LENGTH_M == 0) {
			#pragma unroll
			for (uint32_t k = 0; k < K; k += BLOCKTILE_LENGTH_K) {
				
				// Loading data into shared memory

				// TM iterations for loading A
				for (uint32_t A_i = 0; A_i < BLOCKTILE_AREA_A; A_i += -) {
					AS[(A_threadIdx_Y) * BLOCKTILE_LENGTH_M + A_threadIdx_X]
				}
			}
		}
	}
}
