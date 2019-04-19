#include <iostream>

#include "omp.h"

using namespace std;

int main(int argc, const char* argv) {
	omp_set_num_threads(4);
	omp_set_dynamic(0);

	// cout << omp_get_num_threads() << endl;	
	
#pragma omp parallel
{
	// cout may give corrupted output (i.e. endlines)
	int thread_id = omp_get_thread_num();
	cout << "Thread " << thread_id << " says Hello!" << endl;	
}

	return 0;
}
