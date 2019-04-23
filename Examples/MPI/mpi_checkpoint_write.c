#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int write_checkpoint() {
	// get our rank
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	char file[128];
	sprintf(file, "checkpoint/%d_checkpoint.dat", rank);
	FILE* fp = fopen(file, "w");
	fprintf(fp, "Hello Checkpoint World\n");
	fclose(fp);
	
	return 0;
}
	
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int max_steps = 100;
	int step;
	int checkpoint_every = 10;

	for (step = 0; step < max_steps; step ++) {
		/* perform simulation work */
		if (step % checkpoint_every == 0) {
			write_checkpoint();
		}
	}
	
	MPI_Finalize();

	return 0;
}
