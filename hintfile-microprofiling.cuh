#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

// Not detected with intelliSence
#define syncThreads() __syncthreads()
#define blockIdx_X blockIdx.x
#define blockIdx_Y blockIdx.y
#define threadId threadIdx.x
#define CONSTEVAL_STATIC_ASSERT(c, msg) do { if (!(c)) throw msg; } while(false)
#define __dual__ __device__ __host__

consteval uint32_t arrSizeWithPadding(uint32_t arr_size, uint32_t interval_between_padding) {
	const uint32_t num_rows = arr_size / interval_between_padding;
	return num_rows * (interval_between_padding + 1);
}

consteval bool isPowerOfTwo(uint32_t x) {
	return x && ((x & (x - 1)) == 0);
}

consteval uint8_t LOG2(uint32_t x) {
	uint8_t res = 0;
	while (x > 1) {
		x >>= 1;
		++res;
	}
	return res;
}

consteval uint32_t ABS(int32_t x) {
	return (x > 0) ? x : -x;
}

/// @brief Shifts right if positive otherwise performs a left shift
/// @tparam VAL The amount at which to shift it
/// @param x The value being shifted
/// @return The shifted value of x
template <int32_t VAL>
__dual__ __forceinline uint32_t directionalRightShift(const uint32_t& x) {
	if constexpr (VAL == 0) {
		return x
	}
	else if constexpr (ABS(VAL) > 32) {
		return 0;
	}
	else if constexpr (VAL > 0) {
		return x << VAL;
	}
	else {
		return x >> ABS(VAL);
	}
}

/// @brief Makes the first x bits of a 32 bit unsigned integer 1 and the rest 0
/// @param x The number of 1s
/// @return The resulting unsigned integer
consteval uint32_t genLeftMask(uint8_t x) {
	uint32_t r = 0;
	for (int i = 0; i < x; ++i) {
		r >>= 1;
		r |= (0b0001 << 31);
	}
	return r;
}

template <uint8_t N>
__dual__ __forceinline uint32_t maskLeftBits(const uint32_t& x) {
	return x & genLeftMask(N);
}

/// @brief 
/// @tparam WARP_SIZE The size of a warp
/// @param position 
/// @return 
template <uint32_t WARP_SIZE = 32>
__dual__ __forceinline uint32_t warpSwizzlePosition(const uint32_t& position) noexcept {
	static_assert(isPowerOfTwo(WARP_SIZE), "There needs to be a power of 2 elements per warp");
	
	const uint32_t lane = position & (WARP_SIZE - 1);	// POSITION IN WARP
	const uint32_t row = position >> LOG2(WARP_SIZE);
	const uint32_t rowBase = maskLeftBits<32 - LOG2(WARP_SIZE)>(position);

	return rowBase + (lane ^ (row << 1));
}