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

__global__ void prescan(float* d_out, float* d_in, float* temp, int nGlobe) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = 1;
	int i;
	grid_group g = this_grid();	
	temp[tid << 1] = d_in[tid << 1];
	temp[tid << 1 | 1] = d_in[tid << 1 | 1];

	for (i = (nGlobe >> 1); i > 0; i >>= 1) {
		 g.sync();
		__syncthreads();
		if (tid < i) {
			int from = offset * (tid << 1 | 1) - 1;
			int to = offset * ((tid << 1) + 2) - 1;
			atomicAdd(&temp[to], temp[from]);
		}
		offset <<= 1;
	}
	if (tid == 0) temp[nGlobe - 1] = 0.;
	g.sync();
	__syncthreads();
	for (i = 1; i < nGlobe; i <<= 1) {
		// g.sync();
		 __syncthreads();
		offset >>= 1;
		if (tid < i) {
			int from = offset * (tid << 1 | 1) - 1;
			int to = offset * ((tid << 1) + 2) - 1;
			float swp = temp[from];
			temp[from] = temp[to];
			atomicAdd(&temp[to], swp);
		}
	}
	g.sync();	
	__syncthreads();
	d_out[tid << 1] = temp[tid << 1];
	d_out[tid << 1 | 1] = temp[tid << 1 | 1];
}

int main(int argc, char** argv) {
	const int nDiscretization = 512; // discretization points
	const float highx = 2; // rightmost discretization point
	const int N = 1;
	const int M = 256;
	
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
	for (i = 0; i < nDiscretization; i ++) func_values[i] = 1; 2. * exp(-points[i] * points[i]) * width;
	
	float* d_in;
	float* d_out;
	float* temp;

	cudaMalloc((void**)&d_in, nDiscretization * sizeof(float));
	cudaMalloc((void**)&d_out, nDiscretization * sizeof(float));
	cudaMalloc((void**)&temp, nDiscretization * sizeof(float));

	cudaMemset(d_out, 0, nDiscretization * sizeof(float));

	cudaMemcpy(d_in, func_values,  nDiscretization * sizeof(float), cudaMemcpyHostToDevice);
	prescan<<<N, M>>>(d_out, d_in, temp, nDiscretization);
	
	cudaMemcpy(scan, d_out, nDiscretization * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (i = 0 + nDiscretization - 1; i < nDiscretization; i ++) {
		printf("scan[%d] = %.5f\n", i, scan[i]);
	}
	
	return 0;
}
