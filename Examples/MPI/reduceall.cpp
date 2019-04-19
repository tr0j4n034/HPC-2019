// to compile: mpicxx fileNamze.cpp -o main
// to run: mpirun -np numProcesses ./main

#include <iostream>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank;	
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // identify the rank

	int input = 0;
	if (rank == 0) {
		input = 2;
	} else if (rank == 1) {
		input = 7;
	} else if (rank == 2) {
		input = 2;
	}
	int output;
	
	MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	
	printf("The result is %d at rank %d\n", output, rank);
	
	MPI_Finalize();

	return 0;
}
