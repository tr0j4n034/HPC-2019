// to compile: mpicxx filename.cpp -o main
// to run: mpirun -np num_processes ./main

#include <iostream>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, const char* argv[]) {
	MPI_Init(&argc, &argv);
	printf("Hello World\n");
	MPI_Finalize();
	
	return 0;
}
