// In CUDA we trust.

// To compile & run:
// nvcc gauss_1d_scan.cu -arch=sm_60 -rdc=true -o main && ./main

#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

using namespace cooperative_groups;

// Approximation of Gaussian integral with discretization
// Reference for the integral: https://en.wikipedia.org/wiki/Gaussian_integral
// Mathematically, the correct answer is sqrt(pi). By modifying the parameters, 
// it's possibly to achieve particularly good approximation accuracies.

__global__ void prescan(float* d_in, int nGlobe, int step, int upSweep) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int from = 2 * tid * (step + 1) + step;
	int to = 2 * tid * (step + 1) + 2 * step + 1; 
	if (upSweep) {
		d_in[to] += d_in[from];
	} else {
		int temp = d_in[to];
		d_in[to] += d_in[from];
		d_in[from] = temp;
	}
}
	
int main(int argc, char** argv) {
	const int nDiscretization = 1 << 14; // discretization points
	const float highx = 2; // rightmost discretization point
	
	int N = 1 << 3;
	int M = 1 << 10;
	
	int i;
	float* points = (float*)malloc(nDiscretization * sizeof(float));
	float* func_values = (float*)malloc(nDiscretization * sizeof(float));
	float* scan = (float*)malloc(nDiscretization * sizeof(float));
	float width = 1. / nDiscretization;

	/* 
	 Of course, discretization points can easily be described with an explicit formula.
	 However, to preserve the code readibility, let's keep those elements in the array instead.
	 
	 Since the function is symmetric, we can choose points in [0, highx] and multiply 
    	 approximation value by 2 to get the global function approximation for (-inf, disc_point).
	*/
	
	for (i = 0; i < nDiscretization; i ++) points[i] = 1. * i * highx / nDiscretization;
	for (i = 0; i < nDiscretization; i ++) func_values[i] = 2. * exp(-points[i] * points[i]) * width;
	
	float* d_in;

	cudaMalloc((void**)&d_in, nDiscretization * sizeof(float));
	cudaMemcpy(d_in, func_values,  nDiscretization * sizeof(float), cudaMemcpyHostToDevice);
	
	// Down-sweep phase	
	for (i = 0; i < nDiscretization; i = i * 2 + 1) {
		prescan<<<N, M>>>(d_in, nDiscretization, i, 1);
		if (N > 1) N >>= 1;
		else M >>= 1;
	}
	cudaMemset(&d_in[nDiscretization - 1], 0, sizeof(int));

	N = 1, M = 1;

	// Up-sweep phase
	for ( ; i > 0; i >>= 1) {
		prescan<<<N, M>>>(d_in, nDiscretization, i >> 1, 0);
		if (N < 1024) N <<= 1;
		else M <<= 1;
	}

	cudaMemcpy(scan, d_in, nDiscretization * sizeof(float), cudaMemcpyDeviceToHost);
	
	// for (i = 0; i < nDiscretization; i ++) {
	// 	printf("scan[%d] = %.5f\n", i, scan[i]);
	// }

	printf("%.5f\n", scan[nDiscretization - 1]);
	
	return 0;
}
