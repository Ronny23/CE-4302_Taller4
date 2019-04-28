#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


#define VECTOR_SIZE_1 20000000
#define VECTOR_SIZE_2 30000000
#define VECTOR_SIZE_3 40000000

// Vector definitions
float y_1[VECTOR_SIZE_1], y_2[VECTOR_SIZE_2], y_3[VECTOR_SIZE_3];
float x_1[VECTOR_SIZE_1], x_2[VECTOR_SIZE_2], x_3[VECTOR_SIZE_3];

float a;

// Time variables 
double start_time, run_time;

void saxpy(float *x, float *y, int size)
{

	start_time = omp_get_wtime();

	for (int i = 0; i < size; ++i)
		y[i] = a * x[i] + y[i];

	run_time = omp_get_wtime() - start_time;
	printf("Run time is %f, with %d elements\n", run_time, size);
}


int main()
{
	// Time seed for random numbers
	srand(time(NULL));

	a = rand();

	// Fill vector 1 with random floating point numbers 
	for (int i = 0; i < VECTOR_SIZE_1; ++i)
	{
		x_1[i] = rand();
		y_1[i] = rand();
	}

	// Fill vector 2 with random floating point numbers
	for (int i = 0; i < VECTOR_SIZE_2; ++i){
		x_2[i] = rand();
		y_2[i] = rand();
	}

	// Fill vector 3 with random floating point numbers
	for (int i = 0; i < VECTOR_SIZE_3; ++i)
	{
		x_3[i] = rand();
		y_3[i] = rand();
	}

	// Compute SAXPY serial to vector 1
	saxpy(x_1, y_1, VECTOR_SIZE_1);

	// Compute SAXPY serial to vector 2
	saxpy(x_2, y_2, VECTOR_SIZE_2);

	// Compute SAXPY serial to vector 3
	saxpy(x_3, y_3, VECTOR_SIZE_3);

	return 0;
}