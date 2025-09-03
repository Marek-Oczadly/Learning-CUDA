#pragma once
#include "hintfile-1D.cuh"

#define BLOCKTILED 1


template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKDIM, uint32_t TILESIZE_M = BLOCKDIM, uint32_t TILESIZE_N = TILESIZE_M, 
	uint32_t WARP_SUBTILES_M = 2U, uint32_t WARP_SUBTILES_M = 2U, uint32_t WM = 4U, uint32_t WN = 2U, uint32_t TM = 4U, uint32_t TN = TM, 
	memory_location LOAD_INTO = memory_location::REGISTERS, uint32_t WARPSIZE = 32U>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {

	constexpr uint32_t BLOCKTILE_LENGTH_M = TILESIZE_M * WM;
	constexpr uint32_t BLOCKTILE_LENGTH_N = TILESIZE_N * WN;
	constexpr uint32_t BLOCKTILE_LENGTH_K = WN * TN;

	constexpr uint32_t 

	constexpr uint32_t WARP_SUBTILE_LENGTH_M = ;
	constexpr uint32_t WARP_SUBTILE_LENGTH_M = ;

	constexpr uint32_t NUMTHREADS = BLOCKDIM * BLOCKDIM;

	const uint32_t warpId =		  threadId / WARPSIZE;
	const uint32_t threadInWarp = threadId % WARPSIZE;

	




}