// to compile: mpicxx filename.cpp -o main
// to run: mpirun -np numprocesses ./main
#include <iostream>
#include <stdio.h>

#include <mpi.h>

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("Hello from rank %d out of %d processes in MPI_COMM_WORLD\n", rank, size);
	
	MPI_Finalize();

	return 0;
}
