#include <stdio.h>
#include <math.h>

const int Niter = 5;

__global__ void BlurViaStencil(float* d_out, float* d_in) {
	const float kernel[3][3] = {0.04, 0.12, 0.04,
				    0.12, 0.36, 0.12,
	    			    0.04, 0.12, 0.04};
	int rowID = blockIdx.x + 1;
	int colID = threadIdx.x + 1;
	int pos = rowID * (blockDim.x + 2) + colID;
	d_out[pos] = d_in[pos - blockDim.x - 3] * kernel[0][0]
		     + d_in[pos - blockDim.x - 2] * kernel[0][1]
		     + d_in[pos - blockDim.x - 1] * kernel[0][2]
		     + d_in[pos - 1] * kernel[1][0]
		     + d_in[pos] * kernel[1][1]
		     + d_in[pos + 1] * kernel[1][2]
		     + d_in[pos + blockDim.x + 1] * kernel[2][0]
		     + d_in[pos + blockDim.x + 2] * kernel[2][1]
		     + d_in[pos + blockDim.x + 3] * kernel[2][2];
}

int main(int argc, char** argv) {
	char fileName[] = "img_data.txt";

	freopen(fileName, "r", stdin);	

	int N, M;
	scanf("%d%d", &N, &M);	
	
	int gridSize = (N + 2) * (M + 2);
	int totalBytesSize = gridSize * sizeof(float);

	float* image_old = (float*)malloc(gridSize * sizeof(float));
	float* image_new = (float*)malloc(gridSize * sizeof(float));
	int i, j;

	for (i = 0; i < N; i ++) {
		for (j = 0; j < M; j ++) {
			int pos = (i + 1) * (N + 2) + j + 1;
			scanf("%f", &image_old[pos]); 
		}
	}
	// for (i = 0; i < M + 1; i ++) {
	// 	for (j = 1; j < M + 1; j ++) {
	//		printf("%.5f ", image_old[i * (M + 2) + j]);
	// 	}
	//	printf("\n");
	//}
	//printf("\n\n");

	float* d_in;
	float* d_out;
	
	cudaMalloc((void**)&d_in, totalBytesSize);
	cudaMalloc((void**)&d_out, totalBytesSize);
	
	cudaMemcpy(d_in, image_old, totalBytesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, image_new, totalBytesSize, cudaMemcpyHostToDevice);

	int counter = 0;
	while (counter < Niter) {
		BlurViaStencil<<<N, M>>>(d_out, d_in);
		 BlurViaStencil<<<N, M>>>(d_in, d_out);
		counter += 2;
	}
	cudaMemcpy(image_new, d_in, totalBytesSize, cudaMemcpyDeviceToHost);
	
	FILE* writeFile;
	writeFile = fopen("image_out.txt", "w");
	
	for (i = 1; i < M + 1; i ++) {
		for (j = 1; j < M + 1; j ++) {
			fprintf(writeFile, "%.6f ", image_new[i * (M + 2) + j]);
		}
		fprintf(writeFile, "\n");
	}

	free(image_new);
	free(image_old);

	return 0;
}
