/*
 MonteCarlo PI approximation
 to compile: mpicxx fileName.cpp -o execFile
 to run: mpirun -np numProcesses ./execFile
 Date: 12.04.2019
 Author: Mahmud
*/

#include <iostream>
#include <cstdio>
#include <ctime>
#include <random>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int prank, size;
	int manager = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int TRIALS = 10000;
	int partialSum = 0;
	double x, y;

	default_random_engine generator;
	generator.seed(unsigned(clock()) ^ (prank << 3) ^ (prank + 42));		
	uniform_real_distribution<double> distribution(0., 1.);

	for (int i = 0; i < TRIALS; i ++) {
		double x = distribution(generator);
		double y = distribution(generator);
		if (x * x + y * y < 1.) partialSum ++;		
	}
	int sum = 0;
	MPI_Reduce(&partialSum, &sum, 1, MPI_INT, MPI_SUM, manager, MPI_COMM_WORLD);	
	
	if (prank == manager) {
		double piApprox = 4. * sum / (TRIALS * size);
		printf("Pi is approximated as %.6f\n", piApprox);
	}

	MPI_Finalize();

	return 0;
}
