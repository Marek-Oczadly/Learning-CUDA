#pragma once
#include <cuda_runtime.h>
#include "hintfile-microprofiling.cuh"

/// @brief Loads elements from GMEM into SMEM without vectorised or swizzled loads
/// @tparam BLOCKSIZE The size of the block where the thread is runnuing
/// @tparam NUM_ELEMENTS The number of elements being loaded from GMEM to SMEM
template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS>
__device__ __forceinline void naiveLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[NUM_ELEMENTS]) {
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");

	constexpr uint32_t ELEMENTS_PER_THREAD = NUM_ELEMENTS / BLOCKSIZE;
	constexpr uint32_t LOADS_PER_ITERATION = BLOCKSIZE;

	for (uint32_t i = threadId; i < NUM_ELEMENTS; i += LOADS_PER_ITERATION) {
		SMEM[i] = GMEM[i];
	}
}

/// @brief Loads elements from GMEM into SMEM using vectorised loads
/// @tparam BLOCKSIZE The size of the block where the thread is running. Assumes 1-dimensional blocks
/// @tparam NUM_ELEMENTS The size of the SMEM array
template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS>
__device__ __forceinline void vectorisedLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[NUM_ELEMENTS]) {
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");
	
	constexpr uint32_t ELEMENTS_PER_THREAD = NUM_ELEMENTS / BLOCKSIZE;
	constexpr uint32_t NUM_VECTORISED_LOAD = ELEMENTS_PER_THREAD / 4U;
	constexpr uint32_t LOADS_PER_ITERATION = BLOCKSIZE * 4U;

	for (uint32_t i = threadId * 4; i < NUM_ELEMENTS; i += LOADS_PER_ITERATION) {
		reinterpret_cast<float4*>(&SMEM[i])[0] = reinterpret_cast<const float4*>(&GMEM[i])[0];
	}
}

/// @brief Loads elements from GMEM to SMEM using swizzled and vectorised loads
template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS, uint32_t SWIZZLE_SIZE=32>
__device__ __forceinline void swizzledLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[NUM_ELEMENTS], const uint32_t warpRow, const uint32_t warpCol) {
	// 
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");
	static_assert(isPowerOfTwo(SWIZZLE_SIZE) and (SWIZZLE_SIZE >= 32), "Swizzle size must be a power of 2");
	static_assert(NUM_ELEMENTS % SWIZZLE_SIZE == 0, "The number of elements must be a multiple of the swizzle size");
	// Works in 4 byte words rather than single bytes since the banks are 32 bits wide

	const uint32_t WARP_IDX = threadId >> 5;
	const uint32_t THREAD_IN_WARP = threadId & 31;


	constexpr uint32_t NUM_ROWS = SWIZZLE_SIZE;
	constexpr uint32_t NUM_COLS = NUM_ELEMENTS / NUM_ROWS;

	constexpr uint32_t LOADS_PER_ITERATION = BLOCKSIZE;
	const uint32_t starting_pos = ;	// TODO;

	uint32_t GMEM_POS = threadId;
	for (uint32_t i = starting_pos; i < NUM_ELEMENTS; i += LOADS_PER_ITERATION) {
		SMEM[i] = GMEM[GMEM_POS];
		GMEM_POS += LOADS_PER_ITERATION;
	}

}

template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS, uint32_t SWIZZLE_SIZE = 32>
__device__ __forceinline void swizzledVectorisedLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[NUM_ELEMENTS]) {
	// TODO: Write this function
}

template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS, uint32_t INTERVAL_BETWEEN_PADDING>
__device__ __forceinline void paddedLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[arrSizeWithPadding(NUM_ELEMENTS, INTERVAL_BETWEEN_PADDING)]) {
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");
	
	constexpr uint32_t OUTPUT_ARR_SIZE = arrSizeWithPadding(NUM_ELEMENTS, INTERVAL_BETWEEN_PADDING);
	// TODO: Write this function

}