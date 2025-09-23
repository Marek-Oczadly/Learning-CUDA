#include <cuda_runtime.h>
#include "utils.hpp"
#include "kernel-6a-1D.cuh"
#include <cublas_v2.h>


// Add this helper function at the top of your file after the includes
inline void static checkCudaError(cudaError_t error, const char* function, const char* file, int line) {
	if (error != cudaSuccess) {
		std::cerr << "CUDA error in " << file << ":" << line << " (" << function << ") : "
			<< cudaGetErrorString(error) << std::endl;
		// Ensure all CUDA work is terminated
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

// Macro to make it easier to call the error checker
#define CUDA_CHECK(x) checkCudaError(x, #x, __FILE__, __LINE__)

constexpr bool iscuBLAS = false; // Set to true to use cuBLAS, false to use custom SGEMM kernel

#if defined(_DEBUG)
constexpr bool checkIfWorks = true;	// Set to true to check if the SGEMM works correctly by comparing it with cuBLAS
#else
constexpr bool checkIfWorks = false;
#endif

constexpr uint32_t dim = 2048U;	// Size of the matrices (dim x dim)

int main() {
	constexpr auto fileLocation = "C:/Users/Owner/source/repos/Learning CUDA/Nsight compute/other";
	constexpr bool saveToFile = false; // Set to true to save the output to a file
	constexpr uint32_t M = dim; // Height of A and C
	constexpr uint32_t N = dim; // Width of B and C
	constexpr uint32_t K = dim; // Width of A and Height of B
	constexpr size_t A_size = M * K * sizeof(float);
	constexpr size_t B_size = K * N * sizeof(float);
	constexpr size_t C_size = M * N * sizeof(float);

#if defined(BLOCKTILED)
	constexpr uint32_t BLOCK_SIZE = 16U; // Block size for CUDA kernel
	const dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
	constexpr uint32_t GRID_SIZE_X = CEIL_DIV(N, BLOCK_SIZE * 8);
	constexpr uint32_t GRID_SIZE_Y = CEIL_DIV(M, BLOCK_SIZE * 8);
#elif defined(WARPTILED)
	const dim3 blockDim(128);
	constexpr uint32_t BLOCK_SIZE = 128;
	constexpr uint32_t GRID_SIZE_X = 16;
	constexpr uint32_t GRID_SIZE_Y = 32;
#else
	constexpr uint32_t BLOCK_SIZE = 32U;
	const dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
	constexpr uint32_t GRID_SIZE_X = CEIL_DIV(N, BLOCK_SIZE);
	constexpr uint32_t GRID_SIZE_Y = CEIL_DIV(M, BLOCK_SIZE);
#endif

	const float alpha = 1.0f;
	const float beta = 0.0f;

	float* d_A, * d_B, * d_C;
	cudaMalloc(reinterpret_cast<void**>(&d_A), A_size);
	cudaMalloc(reinterpret_cast<void**>(&d_B), B_size);
	cudaMalloc(reinterpret_cast<void**>(&d_C), C_size);

	float* h_A = generateMatrix(M, K, -10.0f, 10.0f);
	float* h_B = generateMatrix(K, N, -10.0f, 10.0f);
	float* h_C = zeroMatrix<float>(M, N);

	cudaMemcpy(reinterpret_cast<void*>(d_A), h_A, A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(reinterpret_cast<void*>(d_B), h_B, B_size, cudaMemcpyHostToDevice);
	cudaMemcpy(reinterpret_cast<void*>(d_C), h_C, C_size, cudaMemcpyHostToDevice);


	if constexpr(iscuBLAS) {
		cublasHandle_t handle;
		cublasCreate(&handle);

		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
		cudaDeviceSynchronize();
		cublasDestroy(handle);
		std::cout << "cuBLAS SGEMM finished\n";
	}

	#ifdef OneDimensional
		const dim3 griddim(GRID_SIZE_X, GRID_SIZE_Y);
		std::cout << "SGEMM launched with grid size: " << griddim.x << " * " << griddim.y << " and block size: " << blockDim.x << std::endl;
		SGEMM<M, N, K, BLOCK_SIZE> <<<griddim, blockDim >>> (d_A, d_B, d_C);
		CUDA_CHECK(cudaDeviceSynchronize()); // Ensure the kernel has finished executing
		std::cout << "SGEMM finished" << std::endl;
	#endif

	#ifdef TwoDimensional
		const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
		const dim3 griddim(GRID_SIZE_X, GRID_SIZE_Y);
		SGEMM<M, N, K, BLOCK_SIZE> << <griddim, blockDim >> > (d_A, d_B, d_C);
		CUDA_CHECK(cudaDeviceSynchronize()); // Ensure the kernel has finished executing
		std::cout << "SGEMM finished with grid size: " << griddim.x << " * " << griddim.y << " and block size: " << blockDim.x << " * " << blockDim.y << std::endl;
	#endif
	
	// Disable if profiling with nsight
	if constexpr(checkIfWorks) {

		float* h_C1 = new float[M * N];	// Holds the reult of the checked SGEMM
		float* h_C2 = zeroMatrix<float>(M, N); // Holds the result of the naive SGEMM - thorougly checked so I can assume it is correct

		CUDA_CHECK(cudaMemcpy(h_C1, d_C, C_size, cudaMemcpyDeviceToHost)); // Copy the result of the SGEMM to host memory
		CUDA_CHECK(cudaMemcpy(d_C, h_C2, C_size, cudaMemcpyHostToDevice)); // Reset the device memory for the naive SGEMM

		cublasHandle_t handle;
		cublasCreate(&handle);

		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
		cudaDeviceSynchronize();
		cublasDestroy(handle);
		
		std::cout << "cuBLAS SGEMM finished" << std::endl;
		cudaMemcpy(h_C2, d_C, C_size, cudaMemcpyDeviceToHost);

		// Check if they are equal and account for floating point precision
		if constexpr(!saveToFile) {
			if (AreEqualMatrices<M, N>(h_C1, h_C2, 3.0f)) {
				std::cout << "The matrices are equal.\n";
			}
			else {
				std::cout << "The matrices are NOT equal.\n" << "Matrix 1: " << std::endl;
				printMatrix<M, N>(h_C1, 16, 16, 10, 2);
				std::cout << "\nMatrix 2: " << std::endl;
				printMatrix<M, N>(h_C2, 16, 16, 10, 2);
			}
		}
		else {
			fs::path outputPath(fileLocation);
			const std::string baseName = "run_";
			if (!fs::exists(outputPath)) {
				fs::create_directories(outputPath);
			}
			int nextIndex = getNextDirIndex(outputPath, baseName);
			outputPath /= (baseName + std::to_string(nextIndex));
			fs::create_directory(outputPath);
			std::cout << "Saving results to: " << outputPath << std::endl;
			std::ofstream file1(outputPath / "cuBLAS-SGEMM.txt");
			std::ofstream file2(outputPath / "test-SGEMM.txt");
			std::ofstream file3(outputPath / "main.txt");
			std::ofstream file4(outputPath / "matrixA.txt");
			std::ofstream file5(outputPath / "matrixB.txt");

			if (file1.is_open() && file2.is_open() && file3.is_open()) {
				file1 << "cuBLAS SGEMM result:\n";
				printMatrix<M, N>(file1, h_C2, M, N, 10, 2);
				file2 << "SGEMM result:\n";
				printMatrix<M, N>(file2, h_C1, M, N, 10, 2);
				printMatrix<M, K>(file4, h_A, M, K, 10, 2);
				printMatrix<K, N>(file5, h_B, K, N, 10, 2);

				file3 << "Matrix A dimensions: " << M << "x" << K << "\n";
				file3 << "Matrix B dimensions: " << K << "x" << N << "\n";
				file3 << "Matrix C dimensions: " << M << "x" << N << "\n";
				
				if (AreEqualMatrices<M, N>(h_C1, h_C2, 3.0f)) {
					file3 << "The matrices are equal.\n";
				}
				else {
					file3 << "The matrices are NOT equal.\n";
					getDiff<M, N>(file3, h_C1, h_C2, M, N, 3.0f, 8, 2);
				}
			}
			else {
				std::cerr << "Error opening output files.\n";
			}
			file1.close();
			file2.close();
			file3.close();
			file4.close();
			file5.close();
			std::cout << "Results saved to: " << outputPath << std::endl;
		}

		delete[] h_C1;
		delete[] h_C2;

	}

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	// Free device memory
	cudaFree(reinterpret_cast<void*>(d_A));
	cudaFree(reinterpret_cast<void*>(d_B));
	cudaFree(reinterpret_cast<void*>(d_C));

	return 0;
}