#include <random>
#include <iostream>
#include <iomanip>

template<size_t m, size_t n>
void generateMatrix(float (&matrix) [m][n], const float min = 0.0f, const float max = 100.0f) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	for (auto& row : matrix) {
		for (auto& value : row) {
			value = dis(gen);
		}
	}
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