// to compile: mpicxx filename.cpp -o main
// to run: mpirun -np numProcesses ./main

#include <iostream>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, size, len;
	MPI_Init(&argc, &argv);
	char name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &len);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	printf("Hello, world! Process %d of %d on %s\n", rank, size, name);
	
	MPI_Finalize();
	
	return 0;
}
