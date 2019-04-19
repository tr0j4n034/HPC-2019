// to compile: mpicxx fileName.cpp -o main
// to run: mpirun -np numProcesses ./main

#include <iostream>
#include <stdio.h>
#include <stddef.h>

#include <mpi.h>

using namespace std;

typedef struct {
	char ch;
	int max_iter;
	double t0;
	double tf;
	double xmin;
} Pars;

int main(int argc, char** argv) {
	int rank;
	int root = 0; // define the root process 
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // identify the rank

	Pars pars;
	if (rank == 0) {
		pars.ch = 'F';
		pars.max_iter = 10;
		pars.t0 = 0.;
		pars.tf = 1.;
		pars.xmin = -5.;
	}
	
	int nitems = 5; // number of variables inside the struct
	MPI_Datatype types[nitems];
	MPI_Datatype mpi_par; // give my new type a name
	MPI_Aint offsets[nitems]; // an array for storing the element offsets
	int blocklengths[nitems];
	
	types[0] = MPI_CHAR; offsets[0] = offsetof(Pars, ch); blocklengths[0] = 1;
	types[1] = MPI_INT; offsets[1] = offsetof(Pars, max_iter); blocklengths[1] = 1;
	types[2] = MPI_DOUBLE; offsets[2] = offsetof(Pars, t0); blocklengths[2] = 1;
	types[3] = MPI_DOUBLE; offsets[3] = offsetof(Pars, tf); blocklengths[3] = 1;
	types[4] = MPI_DOUBLE; offsets[4] = offsetof(Pars, xmin); blocklengths[4] = 1;

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_par);
	MPI_Type_commit(&mpi_par);

	MPI_Bcast(&pars, 1, mpi_par, root, MPI_COMM_WORLD);
	
	printf("Hello from rank %d: max_iter value is %d\n", rank, pars.max_iter);
	
	if (rank == 0) { // At least, I was curious about these offset values.
		for (int i = 0; i < nitems; i ++) {
			cout << "offsets[" << i << "] = " << offsets[i] << endl;
		}
	}
	MPI_Finalize(); 
	
	return 0;
}

