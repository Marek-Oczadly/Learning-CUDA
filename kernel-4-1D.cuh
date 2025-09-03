#pragma once
#include "hintfile-1D.cuh"

#define BLOCKTILED 1

template <uint32_t M, uint32_t N, uint32_t K,
	uint32_t BLOCKDIM, uint32_t TILESIZE_M = BLOCKDIM, uint32_t TILESIZE_N = TILESIZE_M,
	uint32_t TM = 8U, uint32_t TN = TM, memory_location LOAD_INTO = memory_location::REGISTERS>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {
	// This is mostly the same as kernel-3-1D so the majority of code is copy-pasted from that
	// I'm going to assume that all pointers are at least 16 bit aligned to prevent an unnecessary check

	// Square matrices of equal dimensions
	if constexpr (TILESIZE_M == TILESIZE_N && M == N && TM == TN && LOAD_INTO == memory_location::REGISTERS && TILESIZE_M % 4 == 0 && TILESIZE_N % 4 == 0 && TM % 4 == 0) {


		// Evaluating constexprs. These are optimised at compile time and don't use up a register
		constexpr uint32_t TILESIZE_K = TM;

		constexpr uint32_t BLOCKTILE_LENGTH_M = TILESIZE_M * TM;	// Tile size in the M dimension
		constexpr uint32_t BLOCKTILE_LENGTH_N = TILESIZE_N * TN;	// Tile size in the N dimension
		constexpr uint32_t BLOCKTILE_LENGTH_K = TILESIZE_K;			// Tile size in the K dimension

		constexpr uint32_t BLOCKSIZE = BLOCKDIM * BLOCKDIM;


		__shared__ float AS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_M];
		__shared__ float BS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_N];

		const uint32_t threadRow = threadId % TILESIZE_M;
		const uint32_t threadCol = threadId / TILESIZE_M;

		A += blockIdx_X * BLOCKTILE_LENGTH_M;
		B += blockIdx_Y * BLOCKTILE_LENGTH_N * K;

		float threadResults[TM * TN] = {};


		if constexpr (K % BLOCKTILE_LENGTH_K == 0 && M % BLOCKTILE_LENGTH_M == 0 && BLOCKTILE_AREA_A % BLOCKSIZE == 0 && BLOCKSIZE % BLOCKTILE_LENGTH_M == 0) {
			float regA[TM];
			float regB[TM];

			// Position of thread in each blocktile
			constexpr uint32_t M_OVER_4 = BLOCKTILE_LENGTH_M / 4;	// Allows evluation at compile time
			constexpr uint32_t K_OVER_4 = BLOCKTILE_LENGTH_K / 4;	


			const uint32_t A_threadIdx_X = threadId % M_OVER_4;
			const uint32_t A_threadIdx_Y = threadId / M_OVER_4;

			const uint32_t B_threadIdx_X = threadId % K_OVER_4;
			const uint32_t B_threadIdx_Y = threadId / K_OVER_4;

			constexpr uint32_t STRIDE = 4 * BLOCKSIZE / BLOCKTILE_LENGTH_K;

			for (uint32_t k = 0; k < K; k += BLOCKTILE_LENGTH_K) {

				// Loading data into shared memory
				// Loading A
				#pragma unroll	// Likely very few iterations - 1 or 2 at most
				for (uint32_t i = 0; i < BLOCKTILE_LENGTH_M; i += STRIDE) {
				
					reinterpret_cast<float4*>(&AS[(i + A_threadIdx_Y) * BLOCKTILE_LENGTH_M + 4 * A_threadIdx_X])[0] = 
						reinterpret_cast<const float4*>(&A[(i + A_threadIdx_Y) * M + 4 * A_threadIdx_X])[0];
					
					float4 temp = reinterpret_cast<const float4*>(&B[(i + B_threadIdx_Y) * K + 4 * B_threadIdx_X])[0];
					uint32_t position = 4 * BLOCKTILE_LENGTH_N * B_threadIdx_X + B_threadIdx_Y + i;
					BS[position] = temp.x;
					position += BLOCKTILE_LENGTH_N;
					BS[position] = temp.y;
					position += BLOCKTILE_LENGTH_N;
					BS[position] = temp.z;
					position += BLOCKTILE_LENGTH_N;
					BS[position] = temp.w;	// For some stupid reason w is the last variable in float 4?? Pisses me off
				}

				syncThreads(); // Ensure all data has been loaded into SMEM

				A += BLOCKTILE_LENGTH_K * M;
				B += BLOCKTILE_LENGTH_K;

				for (uint32_t dotIdx = 0; dotIdx < BLOCKTILE_LENGTH_K; ++dotIdx) {
					// Loading values into registers
					{	// Limit the scope of A_pos to free a register
						const uint32_t A_pos = threadRow * TM + BLOCKTILE_LENGTH_M * dotIdx;	// Don't have to compute this value on every iteration
						#pragma unroll
						for (uint32_t TM_i = 0; TM_i < TM; ++TM_i) {
							regA[TM_i] = AS[A_pos + TM_i];
						}
					}
					{	// Limit the scope of B_pos to free a register
						const uint32_t B_pos = threadCol * TN + BLOCKTILE_LENGTH_N * dotIdx;	// Don't have to compute this value on every iteration of i
						#pragma unroll
						for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
							regB[TN_i] = BS[B_pos + TN_i];
						}
					}
					for (uint32_t TM_i = 0; TM_i < TM; ++TM_i) {
						for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
							threadResults[TM_i + TN_i * TM] += regA[TM_i] * regB[TN_i];
						}
					}
				}
				syncThreads();
			}
		}
		// Writing the results into C
		if constexpr (LOAD_INTO == memory_location::REGISTERS) {
			const uint32_t C_Block = blockIdx_X * BLOCKTILE_LENGTH_M + blockIdx_Y * BLOCKTILE_LENGTH_N * M;	// Top left position of the block
			#pragma unroll
			for (uint32_t TM_i = 0; TM_i < TM; TM_i+=4) {
				for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
					
					const uint32_t C_pos = C_Block + (threadRow * TM + TM_i) + M * (TN_i + threadCol * TN);
					const uint32_t results_pos = TM_i + TN_i * TM;

					float4 temp = reinterpret_cast<float4*>(&C[C_pos])[0];

					temp.x = alpha * threadResults[results_pos    ] + beta * temp.x;
					temp.y = alpha * threadResults[results_pos + 1] + beta * temp.y;
					temp.z = alpha * threadResults[results_pos + 2] + beta * temp.z;
					temp.w = alpha * threadResults[results_pos + 3] + beta * temp.w;

					reinterpret_cast<float4*>(&C[C_pos])[0] = temp;
				}
			}
		}
	}

}