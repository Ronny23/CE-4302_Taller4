#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <arm_neon.h> 
#include <time.h>


#define VECTOR_SIZE_1 20000000
#define VECTOR_SIZE_2 30000000
#define VECTOR_SIZE_3 40000000

// Vector definitions with float neon type 
float32_t y_1[VECTOR_SIZE_1], y_2[VECTOR_SIZE_2], y_3[VECTOR_SIZE_3];
float32_t x_1[VECTOR_SIZE_1], x_2[VECTOR_SIZE_2], x_3[VECTOR_SIZE_3];
float32x4_t a;

// Time variables 
double start_time, run_time;

void saxpy_parallel(float *x, float *y, int size)
{

	start_time = omp_get_wtime();

	// OpenMP parallel section
	#pragma omp parallel  
	{
		// Parallel for 
		#pragma omp for
		for (int i = 0; i < size; i+=4){
			float32x4_t y_vec = vld1q_f32(y+i);
			float32x4_t x_vec = vld1q_f32(x+i); 
			// Compute y + a*x 
			y_vec = vmlaq_f32(y_vec, a, x_vec); 
			// Store result in memory
			vst1q_f32(y+i, y_vec); 
		}
	}
	run_time = omp_get_wtime() - start_time;
	printf("Run time is %f, with %d elements\n", run_time, size);
}



int main ()
{
	
	// Time seed for random numbers 
	srand(time(NULL));

	omp_set_num_threads(omp_get_num_procs());
	 
	float32_t cnt = rand();
	a = vdupq_n_f32(cnt);

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

	// Compute SAXPY parallel to vector 1
	saxpy_parallel(x_1, y_1, VECTOR_SIZE_1);

	// Compute SAXPY parallel to vector 2
	saxpy_parallel(x_2, y_2, VECTOR_SIZE_2);

	// Compute SAXPY parallel to vector 3
	saxpy_parallel(x_3, y_3, VECTOR_SIZE_3);

	return 0;
}