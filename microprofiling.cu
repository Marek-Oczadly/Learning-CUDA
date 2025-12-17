#include "hintfile-microprofiling.cuh"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <span>
#include <ranges>
#include <algorithm>

constexpr auto ARR_SIZE = 512 * 16;


inline void static checkCudaError(cudaError_t error, const char* function, const char* file, int line) {
	if (error != cudaSuccess) {
		std::cerr << "CUDA error in " << file << ":" << line << " (" << function << ") : "
			<< cudaGetErrorString(error) << std::endl;
		// Ensure all CUDA work is terminated
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK(x) checkCudaError(x, #x, __FILE__, __LINE__)


int main() {
	const float* const testArrCPU = generateMatrix(1, ARR_SIZE);
	auto pipeline = std::span<const float>(testArrCPU, ARR_SIZE)
		| std::views::transform([](const float x) { return x * 2.0; });
	
	float* const resArrCPU = new float[ARR_SIZE];
	std::ranges::copy(pipeline, resArrCPU);

	float* testArrGPU;
	float* resArrGPU;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&testArrGPU), sizeof(float) * ARR_SIZE));
	CUDA_CHECK(cudaMemcpy(testArrGPU, testArrCPU, ARR_SIZE * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));


}


