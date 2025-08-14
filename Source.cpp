#include "utils.hpp"
#include <iostream>

int main() {
	float testMatrix[4][4];

	generateMatrix(testMatrix);
	printMatrix(testMatrix);
	
	return 0;
}