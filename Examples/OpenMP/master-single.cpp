#include <iostream>
#include <stdio.h>

#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) {
	const int N = 10;
	
	int threadid, i;

#pragma omp parallel 
{
#pragma for private(i)
{
	threadid = omp_get_thread_num();
	for (i = 0; i < N; i ++) {
		printf("Thread %d is saying hello!\n", threadid);		
	}
}
#pragma omp barrier

#pragma omp single
{ // only one thread will enter & execute this block
  // it can be master or any of the other working threads 
	threadid = omp_get_thread_num();
	for (i = 0; i < N; i ++) {
		printf("Thread %d is greeting you!\n", threadid);
	}
}
}

	return 0;
}
