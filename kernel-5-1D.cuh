#pragma once
#include "hintfile-1D.cuh"

#define BLOCKTILED 1


template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKDIM, uint32_t M_N_ratio = 2U, 
		  uint32_t WARP_SUBTILES = 2U, uint32_t WARP_TILES = 2U, uint32_t TM = 4U, uint32_t TN = TM, 
		  memory_location LOAD_INTO = memory_location::REGISTERS, uint32_t WARPSIZE = 32U>
__global__ void SGEMM(const float* __restrict A, const float* __restrict B, float* const __restrict C, const float alpha = 1.0f, const float beta = 0.0f) {

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

	const uint32_t warpId =		  threadId / WARPSIZE;
	const uint32_t threadInWarp = threadId % WARPSIZE;

	float threadResults[TM * TN * WARP_SUBTILES * WARP_SUBTILES] = {};	// Initialise values to 0

	__shared__ float AS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_M];
	__shared__ float BS[BLOCKTILE_LENGTH_K * BLOCKTILE_LENGTH_N];

	A += blockIdx_X * BLOCKTILE_LENGTH_M;
	B += blockIdx_Y * BLOCKTILE_LENGTH_N * K;

	
	




}