#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define OneDimensional 1

// Not detected with intelliSence
#define syncThreads() __syncthreads()
#define blockIdx_X blockIdx.x
#define blockIdx_Y blockIdx.y
#define threadId threadIdx.x

enum class memory_location: uint8_t {
	REGISTERS = 0,
	SHARED_MEM = 1,
	GLOBAL_MEM = 2
};

template <
	uint32_t BLOCKSIZE, uint32_t BLOCKTILE_A_SIZE, uint32_t BLOCKTILE_B_SIZE, uint32_t M, uint32_t N, uint32_t K,
	uint32_t BLOCKTILE_LENGTH_K, uint32_t BLOCKTILE_LENGTH_M, uint32_t BLOCKTILE_LENGTH_N, uint8_t NUM_BUFFERS = 2
>
__device__ __forceinline void loadSMEMTile(
	const float* const __restrict& A, const float* const __restrict& B, float(&AS)[NUM_BUFFERS][BLOCKTILE_A_SIZE], float(&BS)[NUM_BUFFERS][BLOCKTILE_B_SIZE],
	const uint32_t& A_threadIdx_X, const uint32_t& B_threadIdx_X, const uint32_t& A_threadIdx_Y, const uint32_t& B_threadIdx_Y, uint8_t& buffer
) {
	buffer ^= 0x1U;
	// INCLUDES EXTRA BYTE TO SHIFT ALIGNMENT
	constexpr uint32_t SHAREDMEM_LENGTH_M = BLOCKTILE_A_SIZE / BLOCKTILE_LENGTH_K;
	constexpr uint32_t SHAREDMEM_LENGTH_N = BLOCKTILE_B_SIZE / BLOCKTILE_LENGTH_K;

	constexpr uint32_t STRIDE_A = 4 * BLOCKSIZE / BLOCKTILE_LENGTH_M;
	constexpr uint32_t STRIDE_B = 4 * BLOCKSIZE / BLOCKTILE_LENGTH_K;

	#pragma unroll	// Loading A
	for (uint32_t A_i = 0; A_i < BLOCKTILE_LENGTH_K; A_i += STRIDE_A) {
		const uint32_t TIDY = A_i + A_threadIdx_Y;
		reinterpret_cast<float4*>(&AS[buffer][TIDY * SHAREDMEM_LENGTH_M + A_threadIdx_X])[0] =
			reinterpret_cast<const float4*>(&A[TIDY * M + A_threadIdx_X])[0];
	}
	#pragma unroll	// Loading B
	for (uint32_t B_i = 0; B_i < BLOCKTILE_LENGTH_N; B_i += STRIDE_B) {
		const uint32_t TIDY = B_i + B_threadIdx_Y;
		const float4 temp = reinterpret_cast<const float4*>(&B[TIDY * K + B_threadIdx_X])[0];


		uint32_t position = B_threadIdx_X * SHAREDMEM_LENGTH_N + TIDY;

		// transposing B
		BS[buffer][position] = temp.x;
		position += SHAREDMEM_LENGTH_N;
		BS[buffer][position] = temp.y;
		position += SHAREDMEM_LENGTH_N;
		BS[buffer][position] = temp.z;
		position += SHAREDMEM_LENGTH_N;
		BS[buffer][position] = temp.w;	// For some stupid reason w is the last variable in float 4?? Pisses me off
	}
}

template<
	uint32_t BLOCKSIZE, uint32_t BLOCKTILE_A_SIZE, uint32_t BLOCKTILE_B_SIZE, uint32_t M, uint32_t N, uint32_t K,
	uint32_t BLOCKTILE_LENGTH_K, uint32_t BLOCKTILE_LENGTH_M, uint32_t BLOCKTILE_LENGTH_N
>
__device__ __forceinline void loadSMEMTile(
	const float* const __restrict& A, const float* const __restrict& B, float(&AS)[BLOCKTILE_A_SIZE], float(&BS)[BLOCKTILE_B_SIZE],
	const uint32_t& A_threadIdx_X, const uint32_t& B_threadIdx_X, const uint32_t& A_threadIdx_Y, const uint32_t& B_threadIdx_Y
) {
	// INCLUDES EXTRA BYTE TO SHIFT ALIGNMENT
	constexpr uint32_t SHAREDMEM_LENGTH_M = BLOCKTILE_A_SIZE / BLOCKTILE_LENGTH_K;
	constexpr uint32_t SHAREDMEM_LENGTH_N = BLOCKTILE_B_SIZE / BLOCKTILE_LENGTH_K;

	constexpr uint32_t STRIDE_A = 4 * BLOCKSIZE / BLOCKTILE_LENGTH_M;
	constexpr uint32_t STRIDE_B = 4 * BLOCKSIZE / BLOCKTILE_LENGTH_K;

	#pragma unroll	// Loading A
	for (uint32_t A_i = 0; A_i < BLOCKTILE_LENGTH_K; A_i += STRIDE_A) {
		const uint32_t TIDY = A_i + A_threadIdx_Y;
		reinterpret_cast<float4*>(&AS[TIDY * SHAREDMEM_LENGTH_M + A_threadIdx_X])[0] =
			reinterpret_cast<const float4*>(&A[TIDY * M + A_threadIdx_X])[0];
	}

	#pragma unroll	// Loading B
	for (uint32_t B_i = 0; B_i < BLOCKTILE_LENGTH_N; B_i += STRIDE_B) {
		const uint32_t TIDY = B_i + B_threadIdx_Y;
		const float4 temp = reinterpret_cast<const float4*>(&B[TIDY * K + B_threadIdx_X])[0];

		uint32_t position = B_threadIdx_X * SHAREDMEM_LENGTH_N + TIDY;

		// transposing B
		BS[position] = temp.x;
		position += SHAREDMEM_LENGTH_N;
		BS[position] = temp.y;
		position += SHAREDMEM_LENGTH_N;
		BS[position] = temp.z;
		position += SHAREDMEM_LENGTH_N;
		BS[position] = temp.w;	// For some stupid reason w is the last variable in float 4?? Pisses me off
	}
}

template <uint32_t REGFILEDIM_M, uint32_t REGFILEDIM_N, uint32_t TM, uint32_t TN>
__device__ __forceinline void calculate_warptiled_MMA(const float (&regM)[REGFILEDIM_M], const float (&regN)[REGFILEDIM_N], float (&results)[REGFILEDIM_M * REGFILEDIM_N]) {
	for (uint32_t warp_m = 0; warp_m < REGFILEDIM_M; warp_m += TM) {		// Nesting hell
		for (uint32_t warp_n = 0; warp_n < REGFILEDIM_N; warp_n += TN) {
			uint32_t pos = warp_m + warp_n * REGFILEDIM_M;
			for (uint32_t TN_i = 0; TN_i < TN; ++TN_i) {
				#pragma unroll
				for (uint32_t TM_i = 0; TM_i < TM; ++TM_i) {
					results[pos + TM_i] += regM[warp_m + TM_i] * regN[warp_n + TN_i];
				}
				pos += REGFILEDIM_M;
			}
		}
	}
}