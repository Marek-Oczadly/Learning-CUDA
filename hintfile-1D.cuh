#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

enum class memory_location: uint8_t {
	REGISTERS = 0,
	SHARED_MEM = 1,
	GLOBAL_MEM = 2
};

constexpr uint32_t SQUARED(const uint32_t val) {
	return val * val;
}