#include <cuda_runtime.h>
#include "utils.hpp"
#include "naive-kernel.cuh"

int main() {
	constexpr unsigned int M = 4; // Height of A and C
	constexpr unsigned int N = 4; // Width of B and C
	constexpr unsigned int K = 4; // Width of A and Height of B

	float* d_A, * d_B, * d_C;
	{
		constexpr size_t A_size = K * M * sizeof(float);
		float* h_A = generateMatrix(K, M);
		cudaMalloc((void**)&d_A, A_size);
		cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
		delete[] h_A;
	}
	{
		constexpr size_t B_size = K * N * sizeof(float);
		float* h_B = generateMatrix(K, N);
		cudaMalloc((void**)&d_B, B_size);
		cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
		delete[] h_B;
	}
	{
		constexpr size_t C_size = M * N * sizeof(float);
		float* h_C = zeroMatrix<float>(M, N);
		cudaMalloc((void**)&d_C, C_size);
		cudaMemcpy(d_C, h_C, C_size, cudaMemcpyHostToDevice);
		delete[] h_C;
	}
	// Initialize matrices A, B, and C

	const dim3 blockdim(32, 32);
	const dim3 griddim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

	naiveSGEMM<M, N, K> << <griddim, blockdim >> > (d_A, d_B, d_C);


	return 0;
}