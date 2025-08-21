#pragma once
#include <fstream>
#include <string>
#include <random>
#include <filesystem>
#include <regex>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;

constexpr unsigned int CEIL_DIV(const unsigned int a, const unsigned int b) {
	return (a + b - 1) / b;
}

int getNextDirIndex(const fs::path& parentDir, const std::string& baseName) {
	std::regex pattern(baseName + R"(_(\d+))");
	int maxIndex = 0;

	for (const auto& entry : fs::directory_iterator(parentDir)) {
		if (entry.is_directory()) {
			std::smatch match;
			std::string dirname = entry.path().filename().string();
			if (std::regex_match(dirname, match, pattern)) {
				int index = std::stoi(match[1]);
				if (index > maxIndex) maxIndex = index;
			}
		}
	}

	return maxIndex + 1;
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

inline bool isNearlyEqual(const float a, const float b, const float difference = 1.0f) {
	return (std::fabs(a - b) < difference);
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
void printMatrix(const int (&matrix)[m][n], const unsigned char w = 6U) {
	for (const auto& row : matrix) {
		for (const auto& value : row) {
			std::cout << std::setw(w) << value;
		}
		std::cout << '\n';
	}
}

template<size_t m, size_t n>
void printMatrix(const float (&matrix)[m][n], const unsigned char w = 6U, const unsigned char precision = 2U) {
	for (const auto& row : matrix) {
		for (const auto& value : row) {
			std::cout << std::setw(w) << std::fixed << std::setprecision(precision) << value;
		}
		std::cout << '\n';
	}
}

template<size_t M, size_t N>
inline void printMatrix(const float* const matrix, const unsigned int m = M, const unsigned int n = N, const unsigned char w = 6U, const unsigned char precision = 2U) {
	for (unsigned int row = 0; row < m; ++row) {
		for (unsigned int col = 0; col < n; ++col) {
			std::cout << std::setw(w) << std::fixed << std::setprecision(precision) << matrix[col * M + row];
		}
		std::cout << '\n';
	}
}

template <size_t M, size_t N>
inline void printMatrix(std::ofstream& fileStream, const float* const matrix, const unsigned int m = M, const unsigned int n = N, const unsigned char w = 6U, const unsigned char precision = 2U) {
	for (unsigned int row = 0; row < m; ++row) {
		for (unsigned int col = 0; col < n; ++col) {
			fileStream << std::setw(w) << std::fixed << std::setprecision(precision) << matrix[col * M + row];
		}
		fileStream << '\n';
	}
}

template <size_t M, size_t N>
bool AreEqualMatrices(const float* const matA, const float* const matB, const float diff = 0.0f) {
	for (uint32_t i = 0; i < M * N; ++i) {
		if (!isNearlyEqual(matA[i], matB[i], diff)) {
			return false;
		}
	}
	return true;
}

template <size_t M, size_t N>
inline void getDiff(std::ofstream& fileStream, const float* const matA, const float* const matB, const unsigned int m = M, const unsigned int n = N, const float diff = 0.0f, const unsigned char w = 6U, const unsigned char precision = 2U) {
	for (unsigned int row = 0; row < m; ++row) {
		for (unsigned int col = 0; col < n; ++col) {
			if (!isNearlyEqual(matA[col * M + row], matB[col * M + row], diff)) {
				fileStream << std::setw(w) << std::fixed << std::setprecision(precision) << std::fabs(matA[col * M + row] - matB[col * M + row]);
			}
			else {
				fileStream << std::setw(w) << std::fixed << std::setprecision(precision) << 0.0f;
			}
		}
		fileStream << '\n';
	}
}
