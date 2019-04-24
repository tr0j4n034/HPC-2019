#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
	time_t t;
	srand((unsigned) time(&t));

	const int N = 1024 * 4;
	const int M = 1024 * 4;
	const int BINS_COUNT = 15;

	int i;
	int* matrix = (int*)malloc(N * M * sizeof(int));
	int* bins = (int*)malloc(BINS_COUNT * sizeof(int));

	for (i = 0; i < N * M; i ++) {
		matrix[i] = rand() % BINS_COUNT;
	}
	
	double startTime = clock();

	for (i = 0; i < N * M; i ++) bins[matrix[i]] ++;

	double endTime = clock();
	int binSum = 0;
	for (i = 0; i < BINS_COUNT; i ++) {
		printf("The number %d occurs %d times.\n", i, bins[i]);
		binSum += bins[i];
	}
	printf("Sum of bins %d (Expected %d)\n", binSum, N * M);
	printf("\nElapsed time is: %.5f seconds", 1. * (endTime - startTime) / CLOCKS_PER_SEC);	

	return 0;
}
