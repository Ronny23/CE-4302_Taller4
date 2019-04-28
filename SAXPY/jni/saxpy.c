#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <arm_neon.h> 
#include <time.h>


#define VECTOR_SIZE_1 20000000
#define VECTOR_SIZE_2 30000000
#define VECTOR_SIZE_3 40000000

// Vector definitions 
float32_t vec_x1[VECTOR_SIZE_1], vec_x2[VECTOR_SIZE_2], vec_x3[VECTOR_SIZE_3];
float32_t vec_y1[VECTOR_SIZE_1], vec_y2[VECTOR_SIZE_2], vec_y3[VECTOR_SIZE_3];
//float32_t vec_z1[VECTOR_SIZE_1], vec_z2[VECTOR_SIZE_2], vec_z3[VECTOR_SIZE_3];


float32x4_t a_parallel;

float a_serial;

// Time variables 
double start_time, run_time;

void saxpy_serial(float *x, float *y, int size)
{
	start_time = omp_get_wtime();

	for (int i = 0; i < size; ++i)
		y[i] = a_serial * x[i] + y[i];

	run_time = omp_get_wtime() - start_time;
	printf("Run time is %f seconds, with %d elements\n", run_time, size);
}

void saxpy_parallel(float *x_parallel, float *y_parallel, int size)
{
	start_time = omp_get_wtime();

	// OpenMP parallel section 
	#pragma omp parallel  
	{
		// Parallel for 
		#pragma omp for
		for (int i = 0; i < size; i+=4){

			float32x4_t y_vec = vld1q_f32(y_parallel+i); 
			float32x4_t x_vec = vld1q_f32(x_parallel+i); 
			// Compute y + a*x 
			y_vec = vmlaq_f32(y_vec, a_parallel, x_vec); 
			// Store result in memory
			vst1q_f32(y_parallel+i, y_vec); 
		}
	}
	run_time = omp_get_wtime() - start_time;
	printf("Run time is %f seconds, with %d elements\n", run_time, size);
}


int main()
{
	// Time seed for random numbers
	srand(time(NULL));

	a_serial = rand();

	omp_set_num_threads(omp_get_num_procs());

	float32_t cnt = rand();
	a_parallel = vdupq_n_f32(cnt);

	//Pararlell fill of vectors
	#pragma omp parallel  
	{
		#pragma omp for
		for (int i = 0; i < VECTOR_SIZE_1; ++i)
		{
			vec_x1[i] = rand()%100;
			vec_y1[i] = rand()%100;
		}
		#pragma omp for
		for (int i = 0; i < VECTOR_SIZE_2; ++i)
		{
			vec_x2[i] = rand()%100;
			vec_y2[i] = rand()%100;
		}
		#pragma omp for
		for (int i = 0; i < VECTOR_SIZE_3; ++i)
		{
			vec_x3[i] = rand()%100;
			vec_y3[i] = rand()%100;
		}
	}

	printf("\n Compute SAXPY serial \n");
	saxpy_serial(vec_x1,vec_y1,VECTOR_SIZE_1);
	saxpy_serial(vec_x2,vec_y2,VECTOR_SIZE_2);
	saxpy_serial(vec_x3,vec_y3,VECTOR_SIZE_3);


	printf("\n Compute SAXPY parallel \n");
	saxpy_parallel(vec_x1,vec_y1,VECTOR_SIZE_1);
	saxpy_parallel(vec_x2,vec_y2,VECTOR_SIZE_2);
	saxpy_parallel(vec_x3,vec_y3,VECTOR_SIZE_3);



	return 0;
}
