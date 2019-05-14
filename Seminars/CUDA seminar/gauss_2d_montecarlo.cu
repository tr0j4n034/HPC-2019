// In CUDA we trust.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

// Gaussian double integral approximation with CUDA.
// integral approximation can be verified at
// https://www.wolframalpha.com/widgets/gallery/view.jsp?id=f5f3cbf14f4f5d6d2085bf2d0fb76e8a
// For seminar task: f(x, y) = exp(-x^2 - y^2)

float montecarloSerial(float lowx, float highx, float lowy, float highy, int iters) {
	float sum = 0.;
	int i;
	for (i = 0; i < iters; i ++) { // just a for loop, nothing fancy here
		double x = lowx + 1. * rand() / RAND_MAX * (highx - lowx);
		double y = lowy + 1. * rand() / RAND_MAX * (highy - lowy);
		sum += exp(-x * x - y * y);
	}
	return sum / iters * (highx - lowx) * (highy - lowy);
}
__global__ void montecarlo(float* d_out, float __lowx, float __highx,
			float __lowy, float __highy, int __iters) {
	__shared__ float lowx, highx, lowy, highy;
	__shared__ int iters;	

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// let's fix the shared variables for all threads per block once (check the synchronization call).
	if (threadIdx.x == 0) {
		lowx = __lowx, highx = __highx, lowy = __lowy, highy = __highy;
		iters = __iters;
	}
	__syncthreads();

	curandState localState;
	curand_init(tid, 0, 0, &localState);

	int i;
	float x, y, tempSum = 0.;
	for (i = 0; i < iters; i ++) { // each thread calculates its own summation.
		x = lowx + curand_uniform(&localState) * (highx - lowx);
		y = lowy + curand_uniform(&localState) * (highy - lowy);
		tempSum += exp(-x * x - y * y);
	}
	d_out[tid] = tempSum;
}
__global__ void reduce(float* d_out, float* d_in) { // Parallel summation: steps = O(log(N)), work = O(N * log(N))
	extern __shared__ float sdata[];
	
	int globId = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	
	sdata[tid] = d_in[globId];
	__syncthreads();

	int s = blockDim.x >> 1;
	while (s > 0) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
		s >>= 1;
	}
	if (tid == 0) {
		d_out[blockIdx.x] = sdata[0];
	}
}

int main(int argc, char** argv) {
	const int N = 1024;
	const int M = 1024;
	
	// Grid parameters & iteration counts
	const float lowx = -3., highx = +3., lowy = -3., highy = +3.;
	const int perThreadIters = 10;
	const int serialIters = 1000000;

	// Serial approxmiation for the integral
	double approxSerial = montecarloSerial(lowx, highx, lowy, highy, serialIters);
	printf("Serial approximation of integral is %.6f\n", approxSerial);
		
	float* mc_data = (float*)malloc(N * M * sizeof(float));
	float* d_in;
	float* d_intermediate;
	float* d_out;
	
	cudaMalloc((void**)&d_in, N * M * sizeof(float));
	cudaMalloc((void**)&d_intermediate, N * sizeof(float));
	cudaMalloc((void**)&d_out, 1 * sizeof(float));

	// Calculating per thread sums in parallel
	montecarlo<<<N, M>>>(d_in, lowx, highx, lowy, highy, perThreadIters);

	cudaMemcpy(mc_data, d_in, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	// cudaMemset(d_intermediate, 0., N * sizeof(float));
		
	// Reducing the sum in 2 steps:
	reduce<<<N, M, N * sizeof(float)>>>(d_intermediate, d_in);  // (N, M) --> (1, N)
	reduce<<<1, N, N * sizeof(float)>>>(d_out, d_intermediate); // (1, N) --> (1)
	
	float temp;
	cudaMemcpy(&temp, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	
	float approxCuda = temp / (1. * N * M * perThreadIters) * (highx - lowx) * (highy - lowy);	
	printf("Parallel approximation of integral with Cuda is %.6f.\n", approxCuda);

	return 0;
}
