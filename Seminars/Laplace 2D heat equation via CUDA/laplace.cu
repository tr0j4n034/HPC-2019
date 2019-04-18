#include <stdio.h>
#include <math.h>

const int N = 200;
const int M = 200;
const double Niter = 1000;

__global__ void Laplace(float* d_out, float* d_in) {
	int rowID = blockIdx.x + 1;
	int colID = threadIdx.x + 1;
	int pos = rowID * (blockDim.x + 2) + colID;
	d_out[pos] = (d_in[pos - 1] + d_in[pos + 1] +
	       		d_in[pos - blockDim.x - 2] + d_in[pos + blockDim.x + 2]) /  4.;
}

int main(int argc, char** argv) {
	size_t counter = 0;

	FILE* writefile;
	writefile = fopen("out.txt", "w");
	
	const int gridSize = (N + 2) * (M + 2);
	const int ARRAY_BYTES = gridSize * sizeof(float);

	float* T_new = new float[gridSize];
	float* T_old = new float[gridSize];
	
	int i, j;
	for (i = 0; i < gridSize; i ++) {
		T_new[i] = 0;
		T_old[i] = 0;
	}
	for (i = 1; i < M + 1; i ++) {
		T_new[i] = 1.;
		T_old[i] = 1.;
	}
	
	float* d_in;
	float* d_out;

	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);
	
	cudaMemcpy(d_in, T_old, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, T_new, ARRAY_BYTES, cudaMemcpyHostToDevice);

	while (counter < Niter) {
		Laplace<<<N, M>>>(d_out, d_in);
		Laplace<<<N, M>>>(d_in, d_out);	
		counter += 2;
	}
	cudaMemcpy(T_new, d_in, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	for (i = 0; i < M + 1; i ++) {
		for (j = 1; j < M + 1; j ++) {
			fprintf(writefile, "%.6f ", T_new[i * (M + 2) + j]);
		}
		fprintf(writefile, "\n");
	}

	fclose(writefile);

	return 0;
}
