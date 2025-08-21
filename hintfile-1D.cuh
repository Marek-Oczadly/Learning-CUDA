#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

constexpr uint32_t SQUARED(const uint32_t val) {
	return val * val;
}