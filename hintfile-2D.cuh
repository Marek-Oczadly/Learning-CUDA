#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#define TwoDimensional 1

constexpr uint32_t SQUARED(const uint32_t val) {
	return val * val;
}

constexpr uint32_t POWER(const uint32_t base, const uint32_t exp) {
	uint32_t result = 1;
	for (uint32_t i = 0; i < exp; ++i) {
		result *= base;
	}
	return result;
}