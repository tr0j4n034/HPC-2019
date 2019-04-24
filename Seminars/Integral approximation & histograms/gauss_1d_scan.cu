// In CUDA we trust.

#include <stdio.h>
#include <math.h>

// Approximation of Gaussian integral with discretization
// Reference for the integral: https://en.wikipedia.org/wiki/Gaussian_integral
// Mathematically, the correct answer is sqrt(pi). By modifying the parameters, 
// it's possibly to achieve particularly good approximation accuracies.

__global__ void prescan(float* d_out, float* d_in, int nGlobe) {
	extern __shared__ float temp[];
	printf("hello\n");
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = 1;
	int i;
	
	temp[tid << 1] = d_in[tid << 1];
	temp[tid << 1 | 1] = d_in[tid << 1 | 1];
	
	for (i = (nGlobe >> 1); i > 0; i >>= 1) {
		__syncthreads();
		if (tid < i) {
			int from = offset * (tid << 1 | 1) - 1;
			int to = offset * ((tid << 1) + 2) - 1;
			temp[to] += temp[from];
		}
		offset <<= 1;
	}
	if (tid == 0) temp[nGlobe - 1] = 0.;
	for (i = 1; i < nGlobe; i <<= 1) {
		offset >>= 1;
		__syncthreads();
		if (tid < i) {
			int from = offset * (tid << 1 | 1) - 1;
			int to = offset * ((tid << 1) + 2) - 1;
			float swp = temp[from];
			temp[from] = temp[to];
			temp[to] += swp;
		}
	}
	__syncthreads();
	
	d_out[tid << 1] = temp[tid << 1];
	d_out[tid << 1 | 1] = temp[tid << 1 | 1];
}

int main(int argc, char** argv) {
	const int nDiscretization = 1024; // discretization points
	const float highx = 16; // rightmost discretization point
	const int N = 16;
	const int M = 32;
	
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
	for (i = 0; i < nDiscretization; i ++) func_values[i] = 1; // exp(-points[i] * points[i]) * width;

	float* d_in;
	float* d_out;

	cudaMalloc((void**)&d_in, nDiscretization * sizeof(float));
	cudaMalloc((void**)&d_out, nDiscretization * sizeof(float));
	
	cudaMemcpy(d_in, func_values, nDiscretization * sizeof(float), cudaMemcpyHostToDevice);
	printf("going to kernel\n");
	prescan<<<1, 1024, nDiscretization * sizeof(float)>>>(d_out, d_in, nDiscretization);
	
	cudaMemcpy(scan, d_out, nDiscretization * sizeof(float), cudaMemcpyDeviceToHost);

	float maxval = 0.;
	for (i = 0; i < nDiscretization; i ++) {
		printf("%d --> %.5f\n", i, scan[i]);
	}
	
	return 0;
}
