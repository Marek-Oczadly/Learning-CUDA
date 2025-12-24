#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define threadId threadIdx.x
#define CONSTEVAL_STATIC_ASSERT(c, msg) do { if (!(c)) throw msg; } while(false)



consteval uint32_t arrSizeWithPadding(uint32_t arr_size, uint32_t interval_between_padding) {
	CONSTEVAL_STATIC_ASSERT((arr_size % interval_between_padding == 0), "The parameter interval_between_padding must be a fact of arr_size");
	const uint32_t num_rows = arr_size / interval_between_padding;
	return num_rows * (interval_between_padding + 1);
}