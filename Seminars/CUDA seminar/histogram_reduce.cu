// In CUDA we trust.

// When compiling, use -std=c++11 or higher.

#include <stdio.h>
#include <random>
#include <ctime>

__global__ void histogramSimple(int* d_out, const int* d_in, const int BINS_COUNT) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	atomicAdd(&(d_out[d_in[tid] % BINS_COUNT]), 1);
}

int main(int argc, char** argv) {
	srand(unsigned(time(NULL)));

	const int N = 1024;
	const int M = 1024;
	const int BINS_COUNT = 15;

	int i;
	int* matrix = new int[N * M];
	int* bins = new int[BINS_COUNT];
	for (i = 0; i < N * M; i ++) {
		matrix[i] = rand() % BINS_COUNT;
	}
	
	double startTime = clock();
	int* d_in;
	int* d_out;

	cudaMalloc((void**)&d_in, N * M * sizeof(int));
	cudaMalloc((void**)&d_out, BINS_COUNT * sizeof(int));
	
	cudaMemcpy(d_in, matrix, N * M * sizeof(int), cudaMemcpyHostToDevice); 	
	cudaMemcpy(d_out, bins, BINS_COUNT * sizeof(int), cudaMemcpyHostToDevice);
	
	double copyFinishTime = clock();
	histogramSimple<<<N, M>>>(d_out, d_in, BINS_COUNT);
	cudaMemcpy(bins, d_out, BINS_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
	
	double endTime = clock();
	int binSum = 0;
	for (i = 0; i < BINS_COUNT; i ++) {
		printf("The number %d occurs %d times.\n", i, bins[i]);
		binSum += bins[i];
	}
	printf("Sum of bins %d (Expected %d)\n", binSum, N * M);
	printf("\nElapsed time is: %.5f seconds\n", 1. * (endTime - startTime) / CLOCKS_PER_SEC);	
	printf("Elapsed time without data transfer is: %.5f seconds\n", 1. * (endTime - copyFinishTime) / CLOCKS_PER_SEC);
	printf("Data transfer time is: %.5f\n seconds\n", 1. * (copyFinishTime - startTime) / CLOCKS_PER_SEC);
	return 0;
}
