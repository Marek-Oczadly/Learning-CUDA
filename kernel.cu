
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_additon.cuh"

#include <stdio.h>

int main()
{
    float* const a = new float [N];
    float* const b = new float[N];

    float* const b_copy = new float[N];
    RandomGenerator<float> randomGen(0.0f, 100.0f);

    // Generate random vectors
    randomGen.generateRandomVector(a, N);
    randomGen.generateRandomVector(b, N);
    for (int i = 0; i < N; ++i) {
        b_copy[i] = b[i];
    }


    printf("a = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", a[0], a[1], a[2], a[3], a[4], a[N - 5], a[N - 4], a[N - 3], a[N - 2], a[N - 1]);
    printf("b = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", b[0], b[1], b[2], b[3], b[4], b[N - 5], b[N - 4], b[N - 3], b[N - 2], b[N - 1]);


    // Add vectors in parallel.
    cudaError_t cudaStatus = add_inplace(a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("a = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", a[0], a[1], a[2], a[3], a[4], a[N - 5], a[N - 4], a[N - 3], a[N - 2], a[N - 1]);
    printf("b = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", b[0], b[1], b[2], b[3], b[4], b[N - 5], b[N - 4], b[N - 3], b[N - 2], b[N - 1]);

    printf("=======================================================\n");
    printf("a = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", a[0], a[1], a[2], a[3], a[4], a[N - 5], a[N - 4], a[N - 3], a[N - 2], a[N - 1]);
	printf("b = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", b_copy[0], b_copy[1], b_copy[2], b_copy[3], b_copy[4], b_copy[N - 5], b_copy[N - 4], b_copy[N - 3], b_copy[N - 2], b_copy[N - 1]);

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; ++i) {
		b_copy[i] += a[i];
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	printf("CPU execution time: %.5f ms\n", duration.count());
    printf("a = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", a[0], a[1], a[2], a[3], a[4], a[N - 5], a[N - 4], a[N - 3], a[N - 2], a[N - 1]);
	printf("b = {%.2f,%.2f,%.2f,%.2f,%.2f ... %.2f,%.2f,%2.f,%.2f,%.2f}\n", b_copy[0], b_copy[1], b_copy[2], b_copy[3], b_copy[4], b_copy[N - 5], b_copy[N - 4], b_copy[N - 3], b_copy[N - 2], b_copy[N - 1]);
	printf("=======================================================\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] a;
    delete[] b;
	delete[] b_copy;

    return 0;
}