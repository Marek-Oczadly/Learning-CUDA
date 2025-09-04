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
	
	constexpr uint32_t WARP_TILE_LENGTH_M = BLOCKTILE_LENGTH_M / WARP_TILES;
	constexpr uint32_t WARP_TILE_LENGTH_N = BLOCKTILE_LENGTH_N / WARP_TILES;

	constexpr uint32_t WARP_SUBTILE_AREA = WARP_TILE_LENGTH_M * WARP_TILE_LENGTH_N;

	constexpr uint32_t WARP_SUBTILE_LENGTH_M = BLOCKTILE_LENGTH_M / W_SCALE;
	constexpr uint32_t WARP_SUBTILE_LENGTH_N = BLOCKTILE_LENGTH_N / W_SCALE;

	constexpr uint32_t NUMTHREADS = BLOCKDIM * BLOCKDIM;

	constexpr uint32_t M_OVER_4 = BLOCKTILE_LENGTH_M / 4;
	constexpr uint32_t K_OVER_4 = BLOCKTILE_LENGTH_K / 4;

	// WARP CONSTANTS
	const uint32_t warpId		= threadId / WARPSIZE;
	const uint32_t threadInWarp = threadId % WARPSIZE;
	const uint32_t warpIdx_X	= threadId % (WARP_SUBTILE_LENGTH_M / TM);
	const uint32_t warpIdx_Y	= threadId / (WARP_SUBTILE_LENGTH_M / TM);

	float threadResults[TM * TN * WARP_SUBTILES * WARP_SUBTILES] = {};	// Initialise values to 0

	__shared__ float AS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_M];
	__shared__ float BS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_N];

	A += blockIdx_X * BLOCKTILE_LENGTH_M;
	B += blockIdx_Y * BLOCKTILE_LENGTH_N * K;
	
	{	// Calculation

		constexpr uint32_t STRIDE = 4U * BLOCKSIZE / BLOCKTILE_LENGTH_K;

		const uint32_t A_threadIdx_X = 4U * (threadId % M_OVER_4);
		const uint32_t A_threadIdx_Y = threadId / M_OVER_4;

		const uint32_t B_threadIdx_X = 4U * (threadId % K_OVER_4);
		const uint32_t B_threadIdx_Y = threadId / K_OVER_4;

		for (uint32_t k = 0; k < K; k += BLOCKTILE_LENGTH_K) {
			{ // Loading data into SMEM
				#pragma unroll	// Loading A
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

				syncThreads();	// Ensure all data has been loaded into SMEM
			}

			float regA[WARP_SUBTILES * TM];
			float regB[WARP_SUBTILES * TN];
		}
	}
}