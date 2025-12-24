#pragma once
#include "hintfile-microprofiling.cuh"

/// @brief Loads elements from GMEM into SMEM without vectorised or swizzled loads
/// @tparam BLOCKSIZE The size of the block where the thread is runnuing
/// @tparam NUM_ELEMENTS The number of elements being loaded from GMEM to SMEM
template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS>
__device__ __forceinline void naiveLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[NUM_ELEMENTS]) {
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");

	constexpr uint32_t ELEMENTS_PER_THREAD = NUM_ELEMENTS / BLOCKSIZE;
	constexpr uint32_t LOADS_PER_ITERATION = BLOCKSIZE;

	for (uint32_t i = threadId; i < NUM_ELEMENTS; i += BLOCKSIZE) {
		SMEM[i] = GMEM[i];
	}
}

/// @brief Loads elements from GMEM into SMEM using vectorised loads
/// @tparam BLOCKSIZE The size of the block where the thread is running
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
template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS>
__device__ __forceinline void swizzledLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[NUM_ELEMENTS]) {
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");

	constexpr uint32_t SWIZZLE_SIZE = 

}

template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS, uint32_t INTERVAL_BETWEEN_PADDING>
__device__ __forceinline void paddedLoadToSMEM(const float* const __restrict& GMEM, float(&SMEM)[arrSizeWithPadding(NUM_ELEMENTS, INTERVAL_BETWEEN_PADDING)]) {
	static_assert(BLOCKSIZE < 1024, "Blocksize must be less than 1024");
	
	constexpr uint32_t OUTPUT_ARR_SIZE = arrSizeWithPadding(NUM_ELEMENTS, INTERVAL_BETWEEN_PADDING);

}