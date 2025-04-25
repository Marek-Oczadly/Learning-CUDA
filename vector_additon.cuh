#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>

constexpr const unsigned short N = 1024;
constexpr const unsigned short BLOCK_SIZE = 256;

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
	const unsigned short i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

//cudaError_t addWrapper() {
//
//}
