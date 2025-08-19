# Efficient Matrix Multiplication in CUDA

## Theoretical Lower Bounds
* The theoretical lower bound for the number of floating point operations (FLOPs) required to perform SGEMM of Matrices A and B, 
where A is of size **M × K** and B is of size **K × N**, is given by **MN × (2K + 1)**. Given that I am testing 2 equally sized square matrices of size **N × N**, this simplifies to **2N³ + N²**.
The N used in testing is 2048, so the theoretical lower bound is **2*2048³ + 2048² ≈ 17.2 × 10⁹ FLOPs / 17.2 GFLOPs**.
* The lower bound for amount of memory read is **(K × M + N × K + M × N) × sizeof(float)**. Given that I am testing 2 equally sized square matrices of size **N × N** and CUDA uses 4 byte floats,
this simplifies to **12N²**. The N used for testing is 2048, so the lower bound for memory to read is **12 × 2048² ≈ 50.3 MB**.
* At minimum I must write to all values in C with makes the lower bound for memory to write **M × N × sizeof(float)**. Given that M and N are both 4096, this simplifies to  **2048 × 2048 × 4 ≈ 
16.8 MB**.  
* Therefore, there is a total of 67.1MB of memory transfers to and from global memory and 17.2 GFLOPs of computation require to compute the result. 
* The GPU used for testing is an RTX 2060 which has a memory bandwidth of 336 GB/s. 
* The FP32 throughput isn't explicitly specified but Turing architecture can perform FP32 FMA with a latency of 4 cycles. This means that assuming 4 level instruction-level parallelism (ILP)
and full utilisation (which is rarely achieved in practice), each CUDA core can perform 1 FP32 FMA per cycle, or 2 FLOPs/cycle. The RTX 2060 has 1920 CUDA cores and a clock speed of 1680 MHz. 
This means that the theoretical peak FP32 throughput is **1920 × 1680 × 10⁶ × 2 ≈ 6.45 TFLOPs**. This is a theoretical peak and the actual throughput will likely be lower.
* Using these numbers we can calculate that we will need at least **(67.1 × 10⁶) / (336 × 10⁹) ≈ 0.183ms** for the memory transfers and at least **(17.2 × 10⁹) / (6.45 × 10¹²) ≈ 2.66ms** for the computation.
Since computation takes significantly longer that memory transfers, we can assume that <u>**computation will be the bottleneck in this case.**</u>

## Kernel 1 - Naive implementation [(Source code)](./naive-kernel.cuh)
Each thread computes one element of the output Matrix C. For A, the row is held constant and the columns are iterated across. For B, the column is held constant and the rows are iterated across.
### Parameters and variables
#### Template Parameters
* `M`: Height of Matrix A and Matrix C
* `N`: Width of Matrix B and Matrix C
* `K`: Width of Matrix A and Height of Matrix B
#### Function Parameters
* `A`: Pointer to Matrix A with dimensions **M × K**
* `B`: Pointer to Matrix B with dimensions **K × N**
* `C`: Pointer to Matrix C with dimensions **M × N**
* `alpha`: Scalar multiplier for the product of A and B
* `beta`: Scalar multiplier for Matrix C
#### Internal Variables
* `row`: The row of Matrix A and C being processed by the thread
* `col`: The column of Matrix B and C being processed by the thread
* `sum`: Cumulative sum of the row of A and the column of B for the current element of C
* `i`: Loop index that points to the element of the row of A and the column of B being processed
### Nsight Profiling
I used Nsight Compute to profile the kernel. 
## Kernel 2 - One dimensional block

## Sources
* [Proof of number of FLOPs in matrix multiplication](https://math.stackexchange.com/questions/3512976/proof-of-number-of-flops-in-matrix-multiplication)
* [NVidia RTX 2060 Specifications](https://www.nvidia.com/en-gb/geforce/graphics-cards/rtx-2060/)
* [Turing Tuning Guide](https://docs.nvidia.com/cuda/archive//11.7.0/pdf/Turing_Tuning_Guide.pdf)
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)
* [CUTLASS: Fast Linear Algebra in CUDA C++](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)
