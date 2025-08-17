#include <cuda_runtime.h>
#include "utils.hpp"
#include "naive-kernel.cuh"
#include "sgemm-one-dimensional.cuh"

constexpr bool checkIfWorks = true;

int main() {
	constexpr uint32_t M = 4096U; // Height of A and C
	constexpr uint32_t N = 4096U; // Width of B and C
	constexpr uint32_t K = 4096U; // Width of A and Height of B
	constexpr uint32_t BLOCK_SIZE = 32U; // Block size for CUDA kernel
	constexpr uint32_t GRID_SIZE_X = CEIL_DIV(N, BLOCK_SIZE);
	constexpr uint32_t GRID_SIZE_Y = CEIL_DIV(M, BLOCK_SIZE);

	float* d_A, * d_B, * d_C;
	{
		constexpr size_t A_size = K * M * sizeof(float);
		float* h_A = generateMatrix(K, M);
		cudaMalloc(reinterpret_cast<void**>(&d_A), A_size);
		cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
		delete[] h_A;
	}
	{
		constexpr size_t B_size = K * N * sizeof(float);
		float* h_B = generateMatrix(K, N);
		cudaMalloc(reinterpret_cast<void**>(&d_B), B_size);
		cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
		delete[] h_B;
	}
	{
		constexpr size_t C_size = M * N * sizeof(float);
		float* h_C = zeroMatrix<float>(M, N);
		cudaMalloc(reinterpret_cast<void**>(&d_C), C_size);
		cudaMemcpy(d_C, h_C, C_size, cudaMemcpyHostToDevice);
		delete[] h_C;
	}
	// Initialize matrices A, B, and C

#ifdef OneDimensional
	const dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
#else
	const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
#endif
	const dim3 griddim(GRID_SIZE_X, GRID_SIZE_Y);

	OneDimensionalSGEMM<M, N, K, BLOCK_SIZE> <<<griddim, blockDim >>> (d_A, d_B, d_C);
	std::cout << "SGEMM finished with grid size: " << griddim.x << "x" << griddim.y
		<< " and block size: " << blockDim.x << "x" << blockDim.y << std::endl;
	
	// Disable if profiling with nsight
	if constexpr(checkIfWorks) {
		
		constexpr size_t C_size = M * N * sizeof(float);
		float* h_C1 = new float[M * N];	// Holds the reult of the checked SGEMM
		float* h_C2 = zeroMatrix<float>(M, N); // Holds the result of the naive SGEMM - thorougly checked so I can assume it is correct

		cudaMemcpy(h_C1, d_C, C_size, cudaMemcpyDeviceToHost); // Copy the result of the SGEMM to host memory
		cudaMemcpy(d_C, h_C2, C_size, cudaMemcpyHostToDevice); // Reset the device memory for the naive SGEMM

		// Running the naive SGEMM
		const dim3 blockDim2(BLOCK_SIZE, BLOCK_SIZE);
		naiveSGEMM<M, N, K> <<<griddim, blockDim2 >>> (d_A, d_B, d_C);
		
std::cout << "Naive SGEMM finished with grid size: " << griddim.x << "x" << griddim.y
			<< " and block size: " << blockDim2.x << "x" << blockDim2.y << std::endl;
		cudaMemcpy(h_C2, d_C, C_size, cudaMemcpyDeviceToHost);

		// Check if they are equal and account for floating point precision
		if (AreEqualMatrices<M, N>(h_C1, h_C2, 0.001f)) {	
			std::cout << "The matrices are equal.\n";
		} else {	// Print matrices if they are not equal for manual checking
			std:: cout << "The matrices are NOT equal.\n" << "Matrix 1: " << std::endl;
			// Print the first 10x10 elements of each matrix
			printMatrix<M, N>(h_C1, 10, 10, 15, 2);
			std::cout << "\nMatrix 2: " << std::endl;
			printMatrix<M, N>(h_C2, 10, 10, 15, 2);
		}

		delete[] h_C1;
		delete[] h_C2;

	}

	// Free device memory
	cudaFree(reinterpret_cast<void*>(d_A));
	cudaFree(reinterpret_cast<void*>(d_B));
	cudaFree(reinterpret_cast<void*>(d_C));

	return 0;
}