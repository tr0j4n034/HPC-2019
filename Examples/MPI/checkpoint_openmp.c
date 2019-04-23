#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <omp.h>

int main(int argc, char** argv) {
	const int size = 20;
	int nthreads, threadid, i;
	double array1[size], array2[size], array3[size];
	
	// Initialize
	for (i = 0; i < size; i ++) {
		array1[i] = 1. * i;
		array2[i] = 2. * i;
	}
	int chunk = 3;
	
#pragma omp parallel private(threadid)
	{
	threadid = omp_get_thread_num();
	if (threadid == 0) {
		nthreads = omp_get_num_threads();
		printf("Number of threads = %d\n", nthreads);
	}
	printf("My threadid = %d\n", threadid);

#pragma omp for schedule(static, chunk)
	for (i = 0; i < size; i ++) {
		array3[i] = sin(array1[i] + array2[i]);
		printf("Thread id: %d working on index %d\n", threadid, i);
		sleep(1);
	}

	} // join

	printf("TEST array3[199] = %.5f\n", array3[199]); // garbage since 199 >= size	

	return 0;
}
