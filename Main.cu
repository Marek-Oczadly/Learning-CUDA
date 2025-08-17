#include <cuda_runtime.h>
#include "utils.hpp"
#include "naive-kernel.cuh"

int main() {
	constexpr unsigned int M = 6; // Height of A and C
	constexpr unsigned int N = 8; // Width of B and C
	constexpr unsigned int K = 4; // Width of A and Height of B
	constexpr unsigned int BLOCK_SIZE = 32; // Block size for CUDA kernel
	constexpr unsigned int GRID_SIZE_X = CEIL_DIV(N, BLOCK_SIZE);
	constexpr unsigned int GRID_SIZE_Y = CEIL_DIV(M, BLOCK_SIZE);

	float* d_A, * d_B, * d_C;
	{
		constexpr size_t A_size = K * M * sizeof(float);
		float* h_A = generateMatrix(K, M);
		printMatrix<M, K>(h_A, M, K);
		cudaMalloc(reinterpret_cast<void**>(&d_A), A_size);
		cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
		delete[] h_A;
		std::cout << '\n';
	}
	{
		constexpr size_t B_size = K * N * sizeof(float);
		float* h_B = generateMatrix(K, N);
		printMatrix<K, N>(h_B, K, N);
		cudaMalloc(reinterpret_cast<void**>(&d_B), B_size);
		cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
		delete[] h_B;
		std::cout << '\n';
	}
	{
		constexpr size_t C_size = M * N * sizeof(float);
		float* h_C = zeroMatrix<float>(M, N);
		cudaMalloc(reinterpret_cast<void**>(&d_C), C_size);
		cudaMemcpy(d_C, h_C, C_size, cudaMemcpyHostToDevice);
		delete[] h_C;
	}
	// Initialize matrices A, B, and C

	const dim3 blockdim(32, 32);
	const dim3 griddim(GRID_SIZE_X, GRID_SIZE_Y);

	naiveSGEMM<M, N, K> << <griddim, blockdim >> > (d_A, d_B, d_C);

	{
		constexpr size_t C_size = M * N * sizeof(float);
		float* h_C = new float[M * N];
		cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost);
		printMatrix<M, N>(h_C, M, N, 10U, 3U);
		delete[] h_C;
		std::cout << '\n';

	}
	cudaFree(reinterpret_cast<void*>(d_A));
	cudaFree(reinterpret_cast<void*>(d_B));
	cudaFree(reinterpret_cast<void*>(d_C));

	return 0;
}