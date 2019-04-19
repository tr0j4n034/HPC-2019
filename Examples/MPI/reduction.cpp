// to compile: mpicxx fileName.cpp -o main
// to run: mpirun -np numProcesses ./main

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, p, i, root = 0;	

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	// Make the local vector size constant
	int local_vector_size = 100;

	// Compute the global vector size
	int n = p * local_vector_size;

	// initialize the vectors
	double *a, *b;
	
	a = (double*)malloc(local_vector_size * sizeof(double));
	b = (double*)malloc(local_vector_size * sizeof(double));
	
	for (i = 0; i < local_vector_size; i ++) {
		a[i] = 3.14 * rank;
		b[i] = 6.67 * rank;
	}

	// compute the local dot product
	double partial_sum = 0.;
	for (i = 0; i < local_vector_size; i ++) {
		partial_sum += a[i] * b[i];
	}
	
	double sum = 0.;
	MPI_Reduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	if (rank == root) {
		printf("The dot product is %.5f\n", sum);
	}
	free(a);
	free(b);
	
	MPI_Finalize();

	return 0;
}

