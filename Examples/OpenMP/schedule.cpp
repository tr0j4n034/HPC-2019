#include <iostream>
#include <stdio.h>

#include <omp.h>

using namespace std;

int main(int argc, const char* argv[]) {
	const int N = 20;
	
	int nthreads, threadid, i;
	double a[N], b[N], result[N];

	// Initialize
	for (i = 0; i < N; i ++) {
		a[i] = 1. * i;
		b[i] = 2. * i;
	}
	int chunk = 4;

#pragma omp parallel private(threadid)
{ // fork
	threadid = omp_get_thread_num();

#pragma omp for schedule(static, chunk)
	for (i = 0; i < N; i ++) {
		result[i] = a[i] + b[i];
		printf("Thread is: %d working on index %d\n", threadid, i);
	}
} // join

	printf("TEST result[19] = %.5f\n", result[19]);

	return 0;
}
