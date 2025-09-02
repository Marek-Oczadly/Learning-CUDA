#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

// Not detected with intelliSence
#define syncThreads() __syncthreads()
#define blockIdx_X blockIdx.x
#define blockIdx_Y blockIdx.y
#define threadId threadIdx.x

enum class memory_location: uint8_t {
	REGISTERS = 0,
	SHARED_MEM = 1,
	GLOBAL_MEM = 2
};
