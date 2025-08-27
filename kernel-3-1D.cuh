#pragma once
#include "hintfile-1D.cuh"


template <uint32_t M, uint32_t N, uint32_t K, 
		  uint32_t BLOCKSIZE, uint32_t TILESIZE_M, uint32_t TILESIZE_N = TILESIZE_M, 
		  uint32_t TM = 8, uint32_t TN = TM, memory_location LOAD_INTO = memory_location::REGISTERS>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	// Square matrices of equal dimensions
	if constexpr (TILESIZE_M == TILESIZE_N && M == N && TM == TN) {

		// Evaluating constexprs. These are optimised at compile time and don't use up a register
		constexpr uint32_t TILESIZE_K = TM;

		constexpr uint32_t BLOCKTILE_LENGTH_M = TILESIZE_M * TM;	// Tile size in the M dimension
		constexpr uint32_t BLOCKTILE_LENGTH_N = TILESIZE_N * TN;	// Tile size in the N dimension
		constexpr uint32_t BLOCKTILE_LENGTH_K = TILESIZE_K;			// Tile size in the K dimension

		constexpr uint32_t BLOCKTILE_AREA_A = BLOCKTILE_LENGTH_M * BLOCKTILE_LENGTH_K;
		constexpr uint32_t BLOCKTILE_AREA_B = BLOCKTILE_LENGTH_N * BLOCKTILE_LENGTH_K;
		constexpr uint32_t BLOCKTILE_AREA_C = BLOCKTILE_LENGTH_M * BLOCKTILE_LENGTH_N;	// Size of the tile block in C

		__shared__ float AS[BLOCKTILE_LENGTH_M * BLOCKTILE_LENGTH_K];
		__shared__ float BS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_N];


		// Position of thread in each blocktile
		const uint32_t A_threadIdx_X = threadIdx.x % BLOCKTILE_LENGTH_M;
		const uint32_t A_threadIdx_Y = threadIdx.x / BLOCKTILE_LENGTH_M;

		const uint32_t B_threadIdx_X = threadIdx.x % BLOCKTILE_LENGTH_K;
		const uint32_t B_threadIdx_Y = threadIdx.x / BLOCKTILE_LENGTH_K;

		const uint32_t threadRow = ;
		const uint32_t threadCol = 

		A += blockIdx.x * BLOCKTILE_LENGTH_M;
		B += blockIdx.y * BLOCKTILE_LENGTH_N * K;

		float regA[TM];
		float regB[TM];

		if constexpr (LOAD_INTO == memory_location::REGISTERS) {
			float threadResults[TM * TN] = {};
		}

		if constexpr (K % BLOCKTILE_LENGTH_K == 0 && M % BLOCKTILE_LENGTH_M == 0 && BLOCKTILE_AREA_A % BLOCKSIZE == 0 && BLOCKSIZE % BLOCKTILE_LENGTH_M == 0) {
			constexpr uint32_t STRIDE_A = BLOCKSIZE / BLOCKTILE_LENGTH_M;
			constexpr uint32_t STRIDE_B = BLOCKSIZE / BLOCKTILE_LENGTH_K;
			for (uint32_t k = 0; k < K; k += BLOCKTILE_LENGTH_K) {
				
				// Loading data into shared memory
				// Loading A
				#pragma unroll
				for (uint32_t A_i = 0; A_i < BLOCKTILE_LENGTH_K; A_i += STRIDE_A) {
					AS[(A_i + A_threadIdx_Y) * BLOCKTILE_LENGTH_M + A_threadIdx_X] = A[(A_i + A_threadIdx_Y) * M + A_threadIdx_X];
				}

				// Loading B
				#pragma unroll
				for (uint32_t B_i = 0; B_i < BLOCKTILE_LENGTH_N; B_i += STRIDE_B) {
					BS[(B_i + B_threadIdx_Y) * BLOCKTILE_LENGTH_K + B_threadIdx_X] = B[(B_i + B_threadIdx_Y) * K + B_threadIdx_Y];
				}

				__syncthreads(); // Ensure all data has been loaded into SMEM

				A += BLOCKTILE_LENGTH_K * M;
				B += BLOCKTILE_LENGTH_K;

				if constexpr (LOAD_INTO == memory_location::REGISTERS) {
					for (uint32_t dotIdx = 0; dotIdx < BLOCKTILE_LENGTH_K; ++dotIdx) {
						// Loading values into registers
						#pragma unroll
						for (uint32_t TM_i = 0; TM_i < TM; ++TM_i) {
							regA[TM_i] 
						}
						#pragma unroll
						for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
							regB[TN_i]
						}
					}
				}
			}
		}
	}
}
