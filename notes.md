# Efficient Matrix Multiplication in CUDA

## Kernel 1 - Naive implementation
Each thread computers one element of the output Matrix C. For A, the row is held constant and the columns are iterated across. For B, the column is held constant and the rows are iterated across.

## Sources
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)
* [CUTLASS: Fast Linear Algebra in CUDA C++](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)