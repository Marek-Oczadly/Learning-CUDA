#pragma once
#include "hintfile-1D.cuh"

#define WARPTILED 1
//#define BUFFERED 2


#if defined(BUFFERED)
#if !defined(loadBlock)
#define loadBlock() loadSMEMTile<																																	 \
BLOCKSIZE, BLOCKTILE_LENGTH_K* SHAREDMEM_LENGTH_M, SHAREDMEM_LENGTH_N* BLOCKTILE_LENGTH_K, M, N, K, BLOCKTILE_LENGTH_K, BLOCKTILE_LENGTH_M, BLOCKTILE_LENGTH_N>( \
	A, B, AS, BS, A_threadIdx_X, B_threadIdx_X, A_threadIdx_Y, B_threadIdx_Y, buffer_num)
#endif

#if !defined(loadRegisters)
#define loadRegisters(ID, SMEM) loadRegisterFile< \
SHAREDMEM_LENGTH_##ID, WARP_TILE_LENGTH_##ID, WARP_SUBTILE_LENGTH_##ID, T##ID, WARP_SUBTILES, 2, BLOCKTILE_LENGTH_K * SHAREDMEM_LENGTH_##ID>( \
	reg##ID, SMEM, dotIdx, warpIdx_##ID, warpThreadIdx_##ID, buffer_num)
#endif

#else

#if !defined(loadBlock)
#define loadBlock() loadSMEMTile<																																	 \
BLOCKSIZE, BLOCKTILE_LENGTH_K* SHAREDMEM_LENGTH_M, SHAREDMEM_LENGTH_N* BLOCKTILE_LENGTH_K, M, N, K, BLOCKTILE_LENGTH_K, BLOCKTILE_LENGTH_M, BLOCKTILE_LENGTH_N>( \
	A, B, AS, BS, A_threadIdx_X, B_threadIdx_X, A_threadIdx_Y, B_threadIdx_Y)
#endif

#if !defined(loadRegisters)
#define loadRegisters(ID, SMEM) loadRegisterFile< \
SHAREDMEM_LENGTH_##ID, WARP_TILE_LENGTH_##ID, WARP_SUBTILE_LENGTH_##ID, T##ID, WARP_SUBTILES, BLOCKTILE_LENGTH_K * SHAREDMEM_LENGTH_##ID>( \
	reg##ID, SMEM, dotIdx, warpIdx_##ID, warpThreadIdx_##ID)
#endif
#endif


#if !defined(calcResults)
#define calcResults() calculate_warptiled_MMA<THREADDIM_M, THREADDIM_N, TM, TN>(regM, regN, threadResults)
#endif


template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE, uint32_t BLOCKTILE_LENGTH_M = 128, uint32_t BLOCKTILE_LENGTH_N = 64, 
	uint32_t BLOCKTILE_LENGTH_K = 8, uint32_t WARP_SUBTILES = 2U, uint32_t WARP_TILES_M = 2U, uint32_t TM = 4U, uint32_t TN = TM,
	memory_location LOAD_INTO = memory_location::REGISTERS, uint32_t WARPSIZE = 32U>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {

	// BLOCKTILE SIZES
	constexpr uint32_t SHAREDMEM_LENGTH_M = BLOCKTILE_LENGTH_M + 4;
	constexpr uint32_t SHAREDMEM_LENGTH_N = BLOCKTILE_LENGTH_N + 4;


	constexpr uint32_t WARP_TILES_N = (BLOCKSIZE) / (WARPSIZE * WARP_TILES_M);

	// WARP TILE SIZES
	constexpr uint32_t WARP_TILE_LENGTH_M = BLOCKTILE_LENGTH_M / WARP_TILES_M;
	constexpr uint32_t WARP_TILE_LENGTH_N = BLOCKTILE_LENGTH_N / WARP_TILES_N;

	// WARP SUBTILE SIZES
	constexpr uint32_t WARP_SUBTILE_LENGTH_M = WARP_TILE_LENGTH_M / WARP_SUBTILES;
	constexpr uint32_t WARP_SUBTILE_LENGTH_N = WARP_TILE_LENGTH_N / WARP_SUBTILES;


	// WARP CONSTANTS
	const uint32_t warpId = threadId / WARPSIZE;
	const uint32_t threadInWarp = threadId % WARPSIZE;

	// Warp positions
	const uint32_t warpIdx_M = warpId % WARP_TILES_M;
	const uint32_t warpIdx_N = warpId / WARP_TILES_M;

	// Thread positions in warp
	const uint32_t warpThreadIdx_M = threadInWarp % (WARP_SUBTILE_LENGTH_M / TM);
	const uint32_t warpThreadIdx_N = threadInWarp / (WARP_SUBTILE_LENGTH_M / TM);

	// Register dimensions
	constexpr uint32_t THREADDIM_M = TM * WARP_SUBTILES;
	constexpr uint32_t THREADDIM_N = TN * WARP_SUBTILES;

	float threadResults[TM * TN * WARP_SUBTILES * WARP_SUBTILES] = {};	// Initialise values to 0

	#if defined(BUFFERED)
		__shared__ float AS[2][BLOCKTILE_LENGTH_K * SHAREDMEM_LENGTH_M];
		__shared__ float BS[2][BLOCKTILE_LENGTH_K * SHAREDMEM_LENGTH_N];

		uint8_t buffer = 0;
	#else
		__shared__ float AS[BLOCKTILE_LENGTH_K * SHAREDMEM_LENGTH_M];
		__shared__ float BS[BLOCKTILE_LENGTH_K * SHAREDMEM_LENGTH_N];
	#endif

	A += blockIdx_X * BLOCKTILE_LENGTH_M;
	B += blockIdx_Y * BLOCKTILE_LENGTH_N * K;

	{	// Calculation

		constexpr uint32_t M_OVER_4 = BLOCKTILE_LENGTH_M / 4;
		constexpr uint32_t K_OVER_4 = BLOCKTILE_LENGTH_K / 4;


		const uint32_t A_threadIdx_X = 4U * (threadId % M_OVER_4);
		const uint32_t A_threadIdx_Y = threadId / M_OVER_4;

		const uint32_t B_threadIdx_X = 4U * (threadId % K_OVER_4);
		const uint32_t B_threadIdx_Y = threadId / K_OVER_4;


		for (uint32_t k = 0; k < K; k += BLOCKTILE_LENGTH_K) {
			loadBlock();


			syncThreads();	// Ensure all data has been loaded into SMEM

			float regM[WARP_SUBTILES * TM];
			float regN[WARP_SUBTILES * TN];

			for (uint32_t dotIdx = 0; dotIdx < BLOCKTILE_LENGTH_K; ++dotIdx) {

				// LOADING DATA INTO REGISTERS
				// Loading data into regM
				loadRegisters(M, AS);
				loadRegisters(N, BS);

				// Calculating the results
				calcResults();
			}

			A += BLOCKTILE_LENGTH_K * M;
			B += BLOCKTILE_LENGTH_K;
			syncThreads();
		}
	}
	// Writing the results back to C

	constexpr uint32_t THREADTILES_PER_SUBTILE_M = WARP_SUBTILE_LENGTH_M / TM;
	constexpr uint32_t THREADTILES_PER_SUBTILE_N = WARP_SUBTILE_LENGTH_N / TN;

	const uint32_t C_Block = blockIdx_X * BLOCKTILE_LENGTH_M + (blockIdx_Y * BLOCKTILE_LENGTH_N + warpIdx_N * WARP_TILE_LENGTH_N) * M + (warpIdx_M * WARP_TILE_LENGTH_M);
	for (uint32_t warp_m = 0; warp_m < THREADDIM_M; warp_m += TM) {		// Nesting hell
		for (uint32_t warp_n = 0; warp_n < THREADDIM_N; warp_n += TN) {

			const uint32_t C_Warp_Subtile = C_Block + THREADTILES_PER_SUBTILE_M * warp_m + warp_n * M * THREADTILES_PER_SUBTILE_N;
			uint32_t reg_Pos = warp_m + warp_n * THREADDIM_M;

			for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {

				const uint32_t C_pos0 = C_Warp_Subtile + M * (warpThreadIdx_N * TN + TN_i) + warpThreadIdx_M * TM;
				#pragma unroll
				for (uint32_t TM_i = 0; TM_i < TM; TM_i += 4) {

					const uint32_t C_pos = C_pos0 + TM_i;
					const uint32_t results_pos = reg_Pos + TM_i;
					float4 temp = reinterpret_cast<const float4*>(&C[C_pos])[0];

					temp.x = alpha * threadResults[results_pos	  ] + beta * temp.x;
					temp.y = alpha * threadResults[results_pos + 1] + beta * temp.y;
					temp.z = alpha * threadResults[results_pos + 2] + beta * temp.z;
					temp.w = alpha * threadResults[results_pos + 3] + beta * temp.w;

					reinterpret_cast<float4*>(&C[C_pos])[0] = temp;
				}

				reg_Pos += THREADDIM_M;
			}
		}
	}

}