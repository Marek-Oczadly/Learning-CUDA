#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>

constexpr const unsigned short N = 1024;
constexpr const unsigned short BLOCK_SIZE = 256;
constexpr const unsigned short NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

template <typename T>
class RandomGenerator {
private:
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<T> dis;

public:
	RandomGenerator(T min, T max) : gen(rd()), dis(min, max) {}
	T operator()() {
		return dis(gen);
	}
	void generateRandomVector(T* const vector, const size_t size) {
		for (size_t i = 0; i < size; ++i) {
			vector[i] = (*this)();
		}
	}
};

template <typename T>
class RandomIntGenerator {
private:
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_int_distribution<T> dis;

public:
	RandomIntGenerator(T min, T max) : gen(rd()), dis(min, max) {}
	T operator()() {
		return dis(gen);
	}
	void generateRandomVector(T* const vector, const size_t size) {
		for (size_t i = 0; i < size; ++i) {
			vector[i] = (*this)();
		}
	}
};

__global__ void addVector(const float* const A, const float* const B, float* const C, const unsigned short N) {
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

__global__ void addInplace(const float* const A, float* const B, const unsigned short N) {
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		B[i] += A[i];
	}
}


struct StateManager {
	cudaError_t cudaStatus;
	unsigned char byteCode;
	float* gpu_a;
	float* gpu_b;

};

/// @brief Add two vectors in parallel using CUDA.
/// @param input_a The vector to be added to be.
/// @param input_b Added and overwritten
/// @return 
StateManager runInplace(const float* const input_a, float* const input_b, const unsigned short GPU_ID = 0) {
	StateManager state = { cudaSuccess, 0, nullptr, nullptr };

	// Choose which GPU to run on, change this on a multi-GPU system.
	state.cudaStatus = cudaSetDevice(GPU_ID);
	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return state;
	}

	// Allocate GPU buffers for three vectors (two input, one output).
	state.cudaStatus = cudaMalloc(reinterpret_cast<void**>(&state.gpu_a), N * sizeof(float));
	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return state;
	}
	state.byteCode |= 0b00000001;
	
	state.cudaStatus = cudaMalloc(reinterpret_cast<void**>(&state.gpu_b), N * sizeof(float));
	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return state;
	}
	state.byteCode |= 0b00000010;

	state.cudaStatus = cudaMemcpy(state.gpu_a, input_a, N * sizeof(float), cudaMemcpyHostToDevice);
	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return state;
	}

	state.cudaStatus = cudaMemcpy(state.gpu_b, input_b, N * sizeof(float), cudaMemcpyHostToDevice);
	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return state;
	}

	addInplace<<<NUM_BLOCKS, BLOCK_SIZE>>> (state.gpu_a, state.gpu_b, N);
	
	state.cudaStatus = cudaGetLastError();
	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(state.cudaStatus));
		return state;
	}

	state.cudaStatus = cudaMemcpy(input_b, state.gpu_b, N * sizeof(float), cudaMemcpyDeviceToHost);

	if (state.cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return state;
	}
	return state;
}

cudaError_t add_inplace(const float* const input_a, float* const input_b, const unsigned short GPU_ID = 0) {
	StateManager state = runInplace(input_a, input_b, GPU_ID);
	if (state.byteCode & 0b00000001) {
		cudaFree(state.gpu_a);
	}
	if (state.byteCode & 0b00000010) {
		cudaFree(state.gpu_b);
	}
	return state.cudaStatus;
}

