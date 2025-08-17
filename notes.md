# Efficient Matrix Multiplication in CUDA

## Kernel 1 - Naive implementation
Each thread computes one element of the output Matrix C. For A, the row is held constant and the columns are iterated across. For B, the column is held constant and the rows are iterated across.
### Parameters and variables
#### Template Parameters
* `M`: Height if Matrix A and Matrix C
* `N`: Width of Matrix B and Matrix C
* `K`: Width of Matrix A and Height of Matrix B
#### Function Parameters
* `A`: Pointer to Matrix A with dimensions M × K
* `B`: Pointer to Matrix B with dimensions K × N
* `C`: Pointer to Matrix C with dimensions M × N
* `alpha`: Scalar multiplier for the product of A and B
* `beta`: Scalar multiplier for Matrix C
#### Internal Variables
* `row`: The row of Matrix A and C being processed by the thread
* `col`: The column of Matrix B and C being processed by the thread
* `sum`: Cumulative sum of the row of A and the column of B for the current element of C
* `i`: Loop index that points to the element of the row of A and the column of B being processed
### Nsight Profiling
TODO
## Kernel 2 - One dimensional block

## Sources
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)
* [CUTLASS: Fast Linear Algebra in CUDA C++](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)