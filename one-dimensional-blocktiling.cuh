#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE>
__global__ void blocktiling1DSGEMM(const float* A, const float* B, const float* const C, const float alpha, const float beta) {

}