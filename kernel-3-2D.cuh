#pragma once
#include "hintfile-2D.cuh"

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BLOCKSIZE>
__global__ void blocktiling1DSGEMM(const float* A, const float* B, const float* const C, const float alpha, const float beta) {

}