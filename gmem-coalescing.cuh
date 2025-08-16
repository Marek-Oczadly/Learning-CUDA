#pragma once
#include <cuda_runtime.h>

template <unsigned int M, unsigned int N, unsigned int K>
__global__ void coalescedSGEMM(float* A, float* B, float* C, float alpha = 1, float beta = 0) {

}