#include <iostream>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, size, i;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 4) {
		printf("Example is designed for 4 processes!\n");
		MPI_Finalize();
		exit(0);
	}
	
	// A is the send-buffer and B is the receive-buffer
	int A[8], B[4];
	
	// Initialize array
	for (i = 0; i < 8; i ++) {
		A[i] = 0;
		B[i] = 0;
	}
	
	int root = 0; // Define a root process
	
	if (rank == root) {
		// Initialize array A
		A[0] = 3;
		A[1] = 5;
		A[2] = 4;
		A[3] = 1;
		A[4] = 10;
		A[5] = 20;
		A[6] = 30;
		A[7] = 40;
	}

	MPI_Scatter(A, 2, MPI_INT, B, 2, MPI_INT, root, MPI_COMM_WORLD);
	
	printf("Rank %d: B[0] = %d, B[1] = %d, B[2] = %d, B[3] = %d\n",
		rank, B[0], B[1], B[2], B[3]);

	MPI_Finalize();	

	return 0;
}
