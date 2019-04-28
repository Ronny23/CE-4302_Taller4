#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h> 
#include <omp.h>
#include <time.h>

#define VECTOR_SIZE_1 20000000
#define VECTOR_SIZE_2 30000000
#define VECTOR_SIZE_3 40000000

// Vector definitions 
float32_t  vec_x1[VECTOR_SIZE_1], vec_x2[VECTOR_SIZE_2], vec_x3[VECTOR_SIZE_3];
float32_t  vec_y1[VECTOR_SIZE_1], vec_y2[VECTOR_SIZE_2], vec_y3[VECTOR_SIZE_3];

float a = 0;
float32x4_t a_parallel;

double start_time, run_time;

// Product point function without OpenMP
void product_point_serial(float *x, float *y, int size){
	start_time = omp_get_wtime();
	float element = 0;
	for (int i = 0; i < size; ++i)
	{
		element = x[i]*y[i];
		a += element;
	}
	run_time = omp_get_wtime() - start_time;
	printf("Run time is %f seconds, with %d elements\n", run_time, size);
}

//Product point function with OpenMP
void product_point_parallel(float *x_parallel, float *y_parallel, int size){
	
	start_time = omp_get_wtime();
	#pragma omp parallel  
	{
		float32x4_t element_par = vdupq_n_f32(0);
		#pragma omp for private(element_par)
		for (int i = 0; i < size; i+=4)
		{
			float32x4_t y_vec = vld1q_f32(y_parallel+i); 
			float32x4_t x_vec = vld1q_f32(x_parallel+i); 
			element_par = vmulq_f32(x_vec, y_vec);
			a_parallel = vaddq_f32(element_par, a_parallel);
		}
	}
	//Sum acumulator vector
	a = a_parallel[0] + a_parallel[1] + a_parallel[2] + a_parallel[3];
	run_time = omp_get_wtime() - start_time;
	printf("Run time is %f seconds, with %d elements\n", run_time, size);

}



int main (){
	//Initialize vector
	a_parallel = vdupq_n_f32(0);
	
	srand(time(NULL));

	int procs = omp_get_num_procs();

	omp_set_num_threads(procs);
	
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

	printf("\n Running dot product serial \n");
	product_point_serial(vec_x1,vec_y1,VECTOR_SIZE_1);
	product_point_serial(vec_x2,vec_y2,VECTOR_SIZE_2);
	product_point_serial(vec_x3,vec_y3,VECTOR_SIZE_3);

	printf("\n Running dot product parallel \n");
	product_point_parallel(vec_x1,vec_y1,VECTOR_SIZE_1);
	product_point_parallel(vec_x2,vec_y2,VECTOR_SIZE_2);
	product_point_parallel(vec_x3,vec_y3,VECTOR_SIZE_3);

	return 0;
}
