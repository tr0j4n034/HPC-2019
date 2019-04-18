/*
 Parallel quicksort implementation via MPI
 to compile: mpicxx fileName.cpp -o main
 to run: mpirun -np numProcesses ./main
 Date: 12-04-2019
 Created by: Mahmud
*/

#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <climits>
#include <ctime>

#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int size, prank; // process count and process rank
    int tag = 0, countReceived;
    
	MPI_Status status;	

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	const int N = 1000; // number of elements per process
	int data[N]; // elements stored in this array
	
	{
        /*
         Initialize data for each process.
         You can read values from different files or values can be calculated according
         to some formula (or any other means) depending on your problem.
         In this code, the data elements per process are generated via random number generator
        */
        srand(unsigned(clock()) ^ (prank ^ 42) ^ (prank + 42));
        for (int i = 0; i < N; i ++) {
            data[i] = rand() % 100;
        }
	}
    
//    for (int p = 0; p < size; p ++) {
//        if (p == prank) {
//            printf("Initially the process %d has the data elements:\n", prank);
//            for (int i = 0; i < N; i ++) {
//                printf("%d", data[i]);
//                if (i < N - 1) printf(" ");
//            }
//            printf("\n");
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }

    // Real work starts...
    double startTime = MPI_Wtime();
    double endTime;
    
	// Each process sorts its local data (possibly use custom comparator)
	sort(data, data + N);
    
//    for (int p = 0; p < size; p ++) {
//        if (p == prank) {
//            printf("After local sorting the process %d has the data elements:\n", prank);
//            for (int i = 0; i < N; i ++) {
//                printf("%d", data[i]);
//                if (i < N - 1) printf(" ");
//            }
//            printf("\n");
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
    
	int manager = 0; // Define a manager process
    int* splitters = new int[size - 1];
    int* extremePoints = new int[size * size];
    int* buffer = new int[size];
    int exPtr = 0;
    
	MPI_Datatype cutType;
	MPI_Type_vector(size, 1, N / size, MPI_INT, &cutType);
	MPI_Type_commit(&cutType);

	if (prank != manager) {
		int dest = manager;
		MPI_Send(&data[0], 1, cutType, dest, tag, MPI_COMM_WORLD);
	}
	else {
		for (int p = 0; p < size; p ++) {
			if (p == manager) {
                for (int i = N / size - 1; i < N; i += N / size) {
                    extremePoints[exPtr ++] = data[i];
                }
				continue;
			}
			MPI_Recv(&extremePoints[exPtr], size, MPI_INT, p, MPI_ANY_TAG,
				MPI_COMM_WORLD, &status);
            exPtr += size;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
    
    if (prank == manager) {
        sort(extremePoints, extremePoints + exPtr); // take care of extreme points
        for (int ptr = 0; ptr < size - 1; ptr ++) {
            splitters[ptr] = extremePoints[size * (ptr + 1)];
        }
    }
    MPI_Bcast(splitters, size - 1, MPI_INT, manager, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    vector<vector<int> > chunks(size);
    int binID = 0;
    for (int i = 0; i < N; i ++) {
        // int chunkID = int(lower_bound(splitters, splitters + size - 1, data[i]) - splitters);
        if (binID < size - 1 && data[i] > splitters[binID]) binID ++;
        chunks[binID].push_back(data[i]);
    }
    
//    MPI_Barrier(MPI_COMM_WORLD);
//    for (int p = 0; p < size; p ++) {
//        if (p == prank) {
//            printf("After splitting the process %d has the data elements:\n", prank);
//            for (int i = 0; i < size; i ++) {
//                for (int j: chunks[i]) printf("%d ", j);
//                printf("\n");
//            }
//            printf("\n");
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
    
    int finalData[N + 0xab];
    int fSize = 0;
    
    for (int ch = 0; ch < size; ch ++) { // Send each chunk to its final owner
        if (ch == prank) {
            for (int i = 0; i < int(chunks[ch].size()); i ++) {
                finalData[fSize ++] = chunks[ch][i];
            }
            continue;
        }
        int* chunkArray = &chunks[ch][0];
        MPI_Send(&chunkArray[0], int(chunks[ch].size()), MPI_INT, ch, tag, MPI_COMM_WORLD);
    }
    for (int p = 0; p < size; p ++) { // Receive the final chunks
        if (p == prank) continue;
        MPI_Recv(&finalData[fSize], N, MPI_INT, p, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &countReceived);
        fSize += countReceived;
//        for (int i = 0; i < countReceived; i ++) {
//            finalData[fSize ++] = portion[i];
//        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    sort(finalData, finalData + fSize); // Sort the final data
    
//    MPI_Barrier(MPI_COMM_WORLD);
//    for (int p = 0; p < size; p ++) {
//        if (p == prank) {
//            printf("After parallel sorting, the process %d has data elements:\n", prank);
//            for (int i = 0; i < fSize; i ++) {
//                printf("%d ", finalData[i]);
//            }
//            printf("\n");
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (prank == manager) {
        endTime = MPI_Wtime();
        printf("Elapsed time is %.5f seconds\n", endTime - startTime);
    }
    
    delete[] splitters;
    delete[] extremePoints;
    delete[] buffer;
    
    MPI_Finalize();

	return 0;
}
