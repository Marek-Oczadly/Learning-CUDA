#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define threadId threadIdx.x


/// @brief Loads elements from GMEM into SMEM 
/// @tparam BLOCKSIZE The size of the block where the thread is running
/// @tparam NUM_ELEMENTS The size of the SMEM array
template <uint32_t BLOCKSIZE, uint32_t NUM_ELEMENTS>
__device__ __forceinline void loadIntoSMEM(const float* const __restrict &GMEM, float(&SMEM)[NUM_ELEMENTS]) {
	constexpr uint32_t ELEMENTS_PER_THREAD = NUM_ELEMENTS / BLOCKSIZE;
	constexpr uint32_t NUM_VECTORISED_LOAD = ELEMENTS_PER_THREAD / 4U;
	constexpr uint32_t LOADS_PER_ITERATION = BLOCKSIZE * 4U;

	for (uint32_t i = threadId * 4; i < NUM_ELEMENTS; i += LOADS_PER_ITERATION) {
		reinterpret_cast<float4*>(&SMEM[i])[0] = reinterpret_cast<const float4*>(&GMEM[i])[0];
	}
}
