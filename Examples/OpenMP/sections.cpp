#include <iostream>
#include <stdio.h>

#include <omp.h>

using namespace std;

int main(int argc, const char* argv[]) {
	const int N = 100;
	
	int x[N], i, sum, sum2;
	int upper, lower;
	int divide = 20;
	sum = 0;
	sum2 = 0;

#pragma omp parallel for
	for (i = 0; i < N; i ++) {
		x[i] = i;
	}	

#pragma omp parallel private(i) shared(x)
{
	// Fork several different threads
#pragma omp sections
{
	{
		for (i = 0; i < N; i ++) {
			if (x[i] > divide) upper ++;
			if (x[i] <= divide) lower ++;
		}
		printf("The number of points at or below %d in x is %d\n", divide, lower);
		printf("The number of points above %d in x is %d\n", divide, upper);
	}
#pragma omp section
{ // Calculate the sum of x
	for (i = 0; i < N; i ++) {
		sum += x[i];
	}
	printf("Sum of x = %d\n", sum);
}
#pragma omp section
{ // Calculate the sum of squares of x
	for (i = 0; i < N; i ++) {
		sum2 += x[i] * x[i];
	}
	printf("Sum2 of x = %d\n", sum2);
}

}
}

	return 0;
}
