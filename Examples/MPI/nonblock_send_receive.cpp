#include <iostream>
#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int a, b;
	int size, rank;
	int tag = 0; // pick a tag value arbitrarily
	MPI_Status status;
	MPI_Request send_request, recv_request;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (size != 2) {
		printf("Example is designed for 2 processes\n");
		MPI_Finalize();
		exit(0);
	}
	if (rank == 0) {
		a = 314159; // value picked arbitrarily

		MPI_Isend(&a, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(&b, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &recv_request);
		
		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		
		printf("Process %d received value %d\n", rank, b);
	} else {
		a = 667;
		
		MPI_Isend(&a, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(&b, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &recv_request);

		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		
		printf("Process %d received value %d\n", rank, b);
	}

	MPI_Finalize();

	return 0;
}
