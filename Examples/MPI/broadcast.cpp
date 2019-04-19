// to compile: mpicc fileName.cpp -o main
// to run: mpirun -np numProcesses ./main

#include <iostream>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, size, i;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[4];
	
	for (i = 0; i < 4; i ++) {
		A[i] = 0;
	}	

	int root = 0; // Define a root process

	if (rank == root) {
		// Initialize array A
		A[0] = 3;
		A[1] = 5;
		A[2] = 4;
		A[3] = 1;
	}
	MPI_Bcast(A, 4, MPI_INT, root, MPI_COMM_WORLD);

	printf("Rank %d A[0] = %d, A[1] = %d, A[2] = %d, A[3] = %d\n",
		rank, A[0], A[1], A[2], A[3]);

	MPI_Finalize();

	return 0;
}
