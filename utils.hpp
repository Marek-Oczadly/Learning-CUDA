#pragma once
#include <random>
#include <iostream>
#include <iomanip>

constexpr unsigned int CEIL_DIV(const unsigned int a, const unsigned int b) {
	return (a + b - 1) / b;
}

template<size_t m, size_t n>
void generateMatrix(float (&matrix) [m][n], const float min = 0.0f, const float max = 100.0f) noexcept {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	for (auto& row : matrix) {
		for (auto& value : row) {
			value = dis(gen);
		}
	}
}

[[nodiscard]] float* generateSequenceMatrix(const unsigned int m, const unsigned int n) {
	float* const matrix = new float[m * n];
	for (unsigned int i = 0; i < m * n; ++i) {
		matrix[i] = static_cast<float>(i + 1);
	}
	return matrix;
}

[[nodiscard]] float* generateMatrix(const unsigned int m, const unsigned int n, const float min = 0.0f, const float max = 100.0f) noexcept {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	float* const matrix = new float[m * n];
	for (unsigned int i = 0; i < (m * n); ++i) {
		matrix[i] = dis(gen);
	}
	return matrix;
}

template<size_t m, size_t n>
void generateMatrix(int(&matrix)[m][n], const int min = 0, const int max = 100) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(min, max);
	for (auto& row : matrix) {
		for (auto& value : row) {
			value = dis(gen);
		}
	}
}

template<typename T, size_t m, size_t n>
void zeroMatrix(T(&matrix)[m][n]) {
	for (auto& row : matrix) {
		for (auto& value : row) {
			value = T(0);
		}
	}
}

template <typename T>
[[nodiscard]] T* zeroMatrix(const unsigned int m, const unsigned int n) {
	float* const matrix = new float[m * n];
	for (unsigned int i = 0; i < (m * n); ++i) {
		matrix[i] = T(0);
	}
	return matrix;
}

template<size_t m, size_t n>
void printMatrix(const int (&matrix)[m][n], const unsigned char w = 6) {
	for (const auto& row : matrix) {
		for (const auto& value : row) {
			std::cout << std::setw(w) << value;
		}
		std::cout << '\n';
	}
}

template<size_t m, size_t n>
void printMatrix(const float (&matrix)[m][n], const unsigned char w = 6, const unsigned char precision = 2) {
	for (const auto& row : matrix) {
		for (const auto& value : row) {
			std::cout << std::setw(w) << std::fixed << std::setprecision(precision) << value;
		}
		std::cout << '\n';
	}
}

template<size_t M, size_t N>
void printMatrix(const float* const matrix, const unsigned int m = M, const unsigned int n = N, const unsigned char w = 6, const unsigned char precision = 2) {
	for (unsigned int row = 0; row < m; ++row) {
		for (unsigned int col = 0; col < n; ++col) {
			std::cout << std::setw(w) << std::fixed << std::setprecision(precision) << matrix[row * N + col];
		}
		std::cout << '\n';
	}
}