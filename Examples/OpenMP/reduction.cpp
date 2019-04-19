#include <iostream>
#include <stdio.h>

#include <omp.h>

using namespace std;

int main(int argc, const char* argv[]) {
	int i, n, chunk;
	float a[16], b[16], result;
	n = 16;
	chunk = 4;
	result = 0.;
	
	for (i = 0; i < n; i ++) {
		a[i] = 1. * i;
		b[i] = 2. * i;
	}

#pragma omp parallel for default(shared) private(i) schedule(static, chunk) \
	reduction(+: result)
	for (i = 0; i < n; i ++) {
		result += a[i] * b[i];
	}
	
	printf("Result = %.5f\n", result);

	return 0;
}

