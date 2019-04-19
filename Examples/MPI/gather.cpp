// to compile: mpicxx fileName.cpp -o main
// to run: mpirun -np numProcesses ./main

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, size, i;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 4) {
		printf("Example is designed for 4 processes\n");
		MPI_Finalize();
		exit(0);
	}
	
	// A is the send-buffer and B is the receive-buffer
	int A[8], B[8];
	
	// Initialize array
	for (i = 0; i < 8; i ++) {
		A[i] = 0;
		B[i] = 0;
	}
	A[0] = rank;
	A[1] = rank * 11;

	int root = 0; // Define a root process

	MPI_Gather(A, 2, MPI_INT, B, 2, MPI_INT, root, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	printf("Rank %d: ", rank);
	for (i = 0; i < 8; i ++) {
		printf("B[%d] = %d", i, B[i]);
		if (i < 8)	printf(", ");
	}
	printf("\n");

	MPI_Finalize();

	return 0;
}
