// to compile: mpicxx filaName.cpp -o main
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
		printf("This example is designed for 4 processes\n");
		MPI_Finalize();
		exit(0);
	}
	
	int A[8], B[8];
	for (i = 0; i < 4; i ++) {
		A[i << 1] = i + 1 + 10 * rank;
		A[i << 1 | 1] = i + 1 + 100 * rank;  
	}
	
	// Note that the send number and receive number are both one.
	// This reflects the fact that the send size and receive size
	// refer to the distinct data size sent to each process.

	MPI_Alltoall(A, 2,  MPI_INT, B, 2, MPI_INT, MPI_COMM_WORLD);

	printf("Rank: %d ", rank);
	for (i = 0; i < 8; i ++) {
		printf("B[%d] = %d", i, B[i]);
		if (i < 7) printf(", ");
	}
	printf("\n");
	
	MPI_Finalize();
	
	return 0;
}
