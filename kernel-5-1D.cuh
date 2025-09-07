#pragma once
#include "hintfile-1D.cuh"

#define BLOCKTILED 1


template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKDIM, uint32_t M_N_ratio = 2U, 
		  uint32_t WARP_SUBTILES = 2U, uint32_t WARP_TILES = 2U, uint32_t TM = 4U, uint32_t TN = TM, 
		  memory_location LOAD_INTO = memory_location::REGISTERS, uint32_t WARPSIZE = 32U>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {

	// BLOCKTILE SIZES
	constexpr uint32_t BLOCKTILE_LENGTH_M = BLOCKDIM * M_N_ratio * TM;
	constexpr uint32_t BLOCKTILE_LENGTH_N = BLOCKDIM * TN;
	constexpr uint32_t BLOCKTILE_LENGTH_K = TM * WARP_SUBTILES;
	
	// WARP TILE SIZES
	constexpr uint32_t WARP_TILE_LENGTH_M = BLOCKTILE_LENGTH_M / WARP_TILES;
	constexpr uint32_t WARP_TILE_LENGTH_N = BLOCKTILE_LENGTH_N / WARP_TILES;

	// WARP SUBTILE SIZES
	constexpr uint32_t WARP_SUBTILE_LENGTH_M = WARP_TILE_LENGTH_M / WARP_SUBTILES;
	constexpr uint32_t WARP_SUBTILE_LENGTH_N = WARP_TILE_LENGTH_N / WARP_SUBTILES;

	constexpr uint32_t BLOCKSIZE = BLOCKDIM * BLOCKDIM;

	// WARP CONSTANTS
	const uint32_t warpId		= threadId / WARPSIZE;
	const uint32_t threadInWarp = threadId % WARPSIZE;

	// Warp positions
	const uint32_t warpIdx_X	= warpId % WARP_TILES;
	const uint32_t warpIdx_Y	= warpId / WARP_TILES;

	// Thread positions in warp
	const uint32_t warpThreadIdx_X	= threadInWarp % (WARP_SUBTILE_LENGTH_M / TM);
	const uint32_t warpThreadIdx_Y	= threadInWarp / (WARP_SUBTILE_LENGTH_M / TM);

	// Register dimensions
	constexpr uint32_t THREADDIM_M = TM * WARP_SUBTILES;
	constexpr uint32_t THREADDIM_N = TN * WARP_SUBTILES;

	float threadResults[TM * TN * WARP_SUBTILES * WARP_SUBTILES] = {};	// Initialise values to 0

	__shared__ float AS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_M];
	__shared__ float BS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_N];

	A += blockIdx_X * BLOCKTILE_LENGTH_M;
	B += blockIdx_Y * BLOCKTILE_LENGTH_N * K;
	
	{	// Calculation

		constexpr uint32_t M_OVER_4 = BLOCKTILE_LENGTH_M / 4;
		constexpr uint32_t K_OVER_4 = BLOCKTILE_LENGTH_K / 4;


		constexpr uint32_t STRIDE = 4 * BLOCKSIZE / BLOCKTILE_LENGTH_K;


		const uint32_t A_threadIdx_X = 4U * (threadId % M_OVER_4);
		const uint32_t A_threadIdx_Y = threadId / M_OVER_4;

		const uint32_t B_threadIdx_X = 4U * (threadId % K_OVER_4);
		const uint32_t B_threadIdx_Y = threadId / K_OVER_4;

		for (uint32_t k = 0; k < K; k += BLOCKTILE_LENGTH_K) {
			{ // Loading data into SMEM
				//#pragma unroll	// Loading A
				for (uint32_t A_i = 0; A_i < BLOCKTILE_LENGTH_M; A_i += STRIDE) {
					const uint32_t TIDY = A_i + A_threadIdx_Y;
					reinterpret_cast<float4*>(&AS[TIDY * BLOCKTILE_LENGTH_M + A_threadIdx_X])[0] = 
						reinterpret_cast<const float4*>(&A[TIDY * M + A_threadIdx_X])[0];
				}
				
				#pragma unroll	// Loading B
				for (uint32_t B_i = 0; B_i < BLOCKTILE_LENGTH_N; B_i += STRIDE) {
					const uint32_t TIDY = B_i + B_threadIdx_Y;
					float4 temp = reinterpret_cast<const float4*>(&B[TIDY * K + B_threadIdx_X])[0];
					uint32_t position = B_threadIdx_X * BLOCKTILE_LENGTH_N + TIDY;

					// transposing B
					BS[position] = temp.x;
					position += BLOCKTILE_LENGTH_N;
					BS[position] = temp.y;
					position += BLOCKTILE_LENGTH_N;
					BS[position] = temp.z;
					position += BLOCKTILE_LENGTH_N;
					BS[position] = temp.w;	// For some stupid reason w is the last variable in float 4?? Pisses me off
				}
			}

			syncThreads();	// Ensure all data has been loaded into SMEM

			float regM[WARP_SUBTILES * TM];
			float regN[WARP_SUBTILES * TN];
			
			for (uint32_t dotIdx = 0; dotIdx < BLOCKTILE_LENGTH_K; ++dotIdx) {	

				// LOADING DATA INTO REGISTERS
				// Loading data into regM
				#pragma unroll
				for (uint32_t warp_m = 0; warp_m < WARP_SUBTILES; ++warp_m) {
					const uint32_t pos = dotIdx * BLOCKTILE_LENGTH_M + warpIdx_X * WARP_TILE_LENGTH_M + warp_m * WARP_SUBTILE_LENGTH_M + warpThreadIdx_X * TM;
					for (uint32_t i = 0; i < TM; ++i) {
						regM[warp_m * TM + i] = AS[pos + i];
					}
				}

				// Loading data into regN
				#pragma unroll
				for (uint32_t warp_n = 0; warp_n < WARP_SUBTILES; ++warp_n) {
					const uint32_t pos = dotIdx * BLOCKTILE_LENGTH_N + warpIdx_Y * WARP_TILE_LENGTH_N + warp_n * WARP_SUBTILE_LENGTH_N + warpThreadIdx_Y * TN;
					for (uint32_t i = 0; i < TN; ++i) {
						regN[warp_n * TN + i] = BS[pos + i];
					}
				}

				// Calculating the results
				for (uint32_t warp_m = 0; warp_m < THREADDIM_M; warp_m += TM) {		// Nesting hell
					for (uint32_t warp_n = 0; warp_n < THREADDIM_N; warp_n += TN) {
						uint32_t pos = warp_m + warp_n * THREADDIM_M;
						for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
							pos += THREADDIM_M;
							for (uint32_t TM_i = 0; TM_i < TM; ++TM_i) {
								threadResults[pos + TM_i] += regM[warp_m + TM_i] * regN[warp_n + TN_i];
							}
						}
					}
				}
			}

			A += BLOCKTILE_LENGTH_K * M;
			B += BLOCKTILE_LENGTH_K;
			syncThreads();
		}
	}
	// Writing the results back to C

	constexpr uint32_t THREADTILES_PER_SUBTILE_M = WARP_SUBTILE_LENGTH_M / TM;
	constexpr uint32_t THREADTILES_PER_SUBTILE_N = WARP_SUBTILE_LENGTH_N / TN;

	const uint32_t C_Block = blockIdx_X * BLOCKTILE_LENGTH_M + (blockIdx_Y * BLOCKTILE_LENGTH_N + warpIdx_Y * WARP_TILE_LENGTH_N) * M + (warpIdx_X * WARP_TILE_LENGTH_M);
	for (uint32_t warp_m = 0; warp_m < THREADDIM_M; warp_m += TM) {		// Nesting hell
		for (uint32_t warp_n = 0; warp_n < THREADDIM_N; warp_n += TN) {
			const uint32_t C_Warp_Subtile = C_Block + THREADTILES_PER_SUBTILE_M * warp_m + warp_n * K * THREADTILES_PER_SUBTILE_N;
			uint32_t reg_Pos = warp_m + warp_n * THREADDIM_M;
			for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
				const uint32_t C_pos0 = C_Warp_Subtile + TN_i * K;
				reg_Pos += THREADDIM_M;
				for (uint32_t TM_i = 0; TM_i < TM; TM_i += 4) {
					const uint32_t C_pos = C_pos0 + TM_i;
					const uint32_t results_pos = reg_Pos + TM_i;
					float4 temp = reinterpret_cast<const float4*>(&C[C_pos])[0];
					
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