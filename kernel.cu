
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_additon.cuh"

#include <stdio.h>

int main()
{
    float* const a = new float [N];
    float* const b = new float[N];

	float* b_copy = new float[N];
	for (int i = 0; i < N; ++i) {
		b_copy[i] = b[i];
	}

	RandomGenerator<float> randomGen(0.0f, 100.0f);

	// Generate random vectors
	randomGen.generateRandomVector(a, N);
	randomGen.generateRandomVector(b, N);

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

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	delete[] a;
	delete[] b;

    return 0;
}