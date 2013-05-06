#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <defines.h>
#define GIG 1000000000

/***************************************************************************************
* User Definable Values
*/ 
#define CPG 2.52           	// Cycles per GHz -- Adjust to your computer
//#define X_WIDTH 1000 		// must be divisible by 2
//#define Y_WIDTH 1000 		// and sufficiently large (ie > 50)
#define FRAND_MIN -10
#define FRAND_MAX 10
//#define BLOCK_WIDTH 8 		//MUST be a power of 2
#define ITERATIONS 200
#define TOL 0.00
//#define NUM_THREADS 4
/*
* User Definable Values End
***************************************************************************************/


#define OUTPUT_X_WIDTH X_WIDTH/BLOCK_WIDTH
#define OUTPUT_Y_WIDTH Y_WIDTH/BLOCK_WIDTH

#define CALC_INDEX(x,y) (x) * X_WIDTH +(y)

#define CALC_INDEX_MOD(x,y, width) (x) * (width) +(y)
#define GET_TIME(timespec) (int) timespec.tv_sec, (int) timespec.tv_nsec
//#define RECORD_START clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
//#define RECORD_END clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

#define RECORD_START current_utc_time(&start);
#define RECORD_END current_utc_time(&end);
#define CHECK_CORRECTNESS check_downsample(check, output);

float * init_matrix(int x_width, int y_width);
struct timespec diff(struct timespec, struct timespec);
double fRand(double fmin, double fmax);
void seed_matrix(float * matrix, int size);
int log_2(int number);
//int clock_gettime(clockid_t clk_id, struct timespec *tp);
void current_utc_time(struct timespec *ts);
void * thread_ds(void * arg);
void check_downsample(float * ref, float * out);


/***************************************************************************************
* Testing Functions
*/
void downsample(const float * matrix, float * output);
void downsample_tiered(float * input, float * output);
void smooth(const float * input, float * output);
void threaded_downsample(const float * input, float * output, int num_of_threads);
/*
*
***************************************************************************************/


main(int argc, char * argv[]) {

   float * matrix, *output, *check;
   struct timespec start, end, difference;
   long int loop;

/**************************************************************************************
* Input Initialization
*/
   matrix = init_matrix(X_WIDTH, Y_WIDTH);
   srand(1);
   seed_matrix(matrix, X_WIDTH*Y_WIDTH);
   check = init_matrix(OUTPUT_X_WIDTH, OUTPUT_Y_WIDTH);
   downsample(matrix, check);
/*
* Initialization End
***************************************************************************************/

/**************************************************************************************
* Downsampling Test
*/
	output = init_matrix(OUTPUT_X_WIDTH, OUTPUT_Y_WIDTH);

	RECORD_START
        for (loop = ITERATIONS; loop > 0; loop--) {
		downsample(matrix, output);   
	}
	RECORD_END
	difference = diff(start, end);
	CHECK_CORRECTNESS
	free(output);

//	printf("Time to do %d iterations       downsample %d x %d by a factor of %d: \t%d.%09d s\n", ITERATIONS, X_WIDTH, Y_WIDTH, BLOCK_WIDTH, GET_TIME(difference));
	printf("%d.%d \t",GET_TIME(difference));
/*
* Downsampling Test End
***************************************************************************************/

/**************************************************************************************
* Tiered Downsampling Test
*/
	output = init_matrix(OUTPUT_X_WIDTH, OUTPUT_Y_WIDTH);
   
   	RECORD_START
	for (loop = ITERATIONS; loop > 0; loop--) {
		downsample_tiered(matrix, output);
	}
	RECORD_END
	difference = diff(start, end);
	CHECK_CORRECTNESS
	free(output);
   
//   	printf("Time to do %d iterations tiered downsample %d x %d by a factor of %d: \t%d.%09d s\n", ITERATIONS, X_WIDTH, Y_WIDTH, BLOCK_WIDTH, GET_TIME(difference));      
	printf("%d.%d\n", GET_TIME(difference));
/*
* Tiered Downsampling Test
***************************************************************************************/   
/**************************************************************************************
* Threaded Downsampling Test
*/
/*	output = init_matrix(OUTPUT_X_WIDTH, OUTPUT_Y_WIDTH);
   
   	RECORD_START
	for (loop = ITERATIONS; loop > 0; loop--) {
		threaded_downsample(matrix, output, NUM_THREADS);
	}
	RECORD_END
	difference = diff(start, end);
	CHECK_CORRECTNESS
	free(output);
   
   	printf("Time to do %d iterations %d-threaded downsample %d x %d by a factor of %d: \t%d.%09d s\n", ITERATIONS, NUM_THREADS, X_WIDTH, Y_WIDTH, BLOCK_WIDTH, GET_TIME(difference));      
*/
/*
* Threaded Downsampling Test
***************************************************************************************/   

/***************************************************************************************
* Smoothing Test
*/
/*
	output = init_matrix(X_WIDTH, Y_WIDTH);

	RECORD_START	
	smooth(matrix, output);
	RECORD_END
	difference = diff(start, end);
	free(output);

	printf("Time to smooth matrix of size %d x %d: \t\t\t%d.%09d s\n", X_WIDTH, Y_WIDTH, GET_TIME(difference));
*/
/*
* Smoothing Test End
***************************************************************************************/   

/***************************************************************************************
* Cleanup
*/
   	free(matrix);
/*
* Cleanup End
***************************************************************************************/
}


/*************************************************************************************
* Helper Functions
*/
float * init_matrix(int x_width, int y_width) {
	float *temp = calloc(x_width *y_width, sizeof(float));
    return temp;
}

void seed_matrix(float * matrix, int size) {
  	int i;
  	for (i=0; i < size; i++) {
		matrix[i] = fRand(FRAND_MIN, FRAND_MAX);
  	}
}

struct timespec diff(struct timespec start, struct timespec end)
{
  	struct timespec temp;
  	if ((end.tv_nsec-start.tv_nsec)<0) {
    	temp.tv_sec = end.tv_sec-start.tv_sec-1;
    	temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  	} else {
  		temp.tv_sec = end.tv_sec-start.tv_sec;
   	 	temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  	}
  	return temp;
}

double fRand(double fMin, double fMax)
{
  	double f = (double)random() / (double)(RAND_MAX);
  	return fMin + f * (fMax - fMin);
}

int log_2(int num) {
	switch (num) {
		case 1   : return 0;
		case 2   : return 1;
		case 4   : return 2;
		case 8   : return 3;
		case 16  : return 4;
		case 32  : return 5;
		case 64  : return 6;
		case 128 : return 7;
		case 256 : return 8;
		case 512 : return 9;
		case 1024: return 10;
	}
}


void current_utc_time(struct timespec *ts) {
 
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ts);
#endif
 
}

void check_downsample(float * ref, float * out) {
	int i, j;
	int count = 0;

	for (i = 0; i < OUTPUT_X_WIDTH; i++) {
		for (j = 0; j < OUTPUT_Y_WIDTH; j++) {
			if(abs(ref[CALC_INDEX_MOD(i,j,OUTPUT_X_WIDTH)] - out[CALC_INDEX_MOD(i,j,OUTPUT_X_WIDTH)]) > TOL) {
				count++;
			}
		}
	}

	if(count > 0) {
		printf("Invalid Operations: %d\n", count);
	}
}
/*
* Helper Functions End
***************************************************************************************/
/*************************************************************************************
* Testing Functions 
*/

/*
* Function for a simple downsampling by a factor of BLOCK_WIDTH^2
*
* Replaces every BLOCK_WIDTH X BLOCK_WIDTH square with the average
* of the squares elements 
*/
void downsample(const float * matrix, float * output) {
	int x,y, xx, yy;
   
   	for (x = 0; x < X_WIDTH; x+= BLOCK_WIDTH) {
		for (y = 0; y < Y_WIDTH; y += BLOCK_WIDTH) {
			float sum = 0.0f;
 	   		for (xx = x; xx < x+BLOCK_WIDTH; xx++) {
				for (yy = y; yy < y+BLOCK_WIDTH; yy++) {
		 			sum += matrix[CALC_INDEX(xx, yy)];
				}
	   		}
	   		output[CALC_INDEX_MOD(x/BLOCK_WIDTH, y/BLOCK_WIDTH, OUTPUT_X_WIDTH)] = sum / (BLOCK_WIDTH*BLOCK_WIDTH);
		}
   	}
}

typedef struct {
	int start;
	int stride;
	const float * input;
	float * output;
} thread_data_t;

void threaded_downsample(const float * input, float * output, int num_of_threads) {
	int i;
	pthread_t ids[num_of_threads];
	thread_data_t data[num_of_threads];
	int stride = num_of_threads * BLOCK_WIDTH;
	
	for (i = 0; i < num_of_threads; i++) {
		data[i].start = i * BLOCK_WIDTH;
		data[i].stride = stride;
		data[i].input = input;
		data[i].output = output;
		pthread_create(&ids[i], NULL, thread_ds, (void *) &data[i]);
	}

	for (i = 0; i < num_of_threads; i++){
		pthread_join(ids[i], NULL);
	}
}

void * thread_ds(void * arg) {
	int x, y, xx, yy;
	thread_data_t * data = (thread_data_t *) arg;
	int start = data->start;
	int stride = data->stride;
	const float * input = data->input;
	float * output = data->output;

	for (x = start; x < X_WIDTH; x+= stride) {
                for (y = 0; y < Y_WIDTH; y += BLOCK_WIDTH) {
                        float sum = 0.0f;
                        for (xx = x; xx < x+BLOCK_WIDTH; xx++) {
                                for (yy = y; yy < y+BLOCK_WIDTH; yy++) {
                                        sum += input[CALC_INDEX(xx, yy)];
                                }
                        }
                        output[CALC_INDEX_MOD(x/BLOCK_WIDTH, y/BLOCK_WIDTH, OUTPUT_X_WIDTH)] = sum / (BLOCK_WIDTH*BLOCK_WIDTH);
                }
        }

}
	

/*
* Function for doing a Tiered and iterative downsampling
*
* This function achieves a downsampling rate of BLOCK_WIDTH ^ 2
* by iteratively downsampling by a factor of 2 in each direction
* for log_2(BLOCK_WIDTH) iterations.
*
* Ex. BLOCK_WIDTH = 16 -> 4 iterations
* 			Iteration 1: Downsample [ original ]  by 2 (factor of  2 overall) store into [  temp  ]
* 			Iteration 2: Downsample [   temp   ]  by 2 (factor of  4 overall) store into [  temp  ]
* 			Iteration 3: Downsample [   temp   ]  by 2 (factor of  8 overall) store into [  temp  ]
* 			Iteration 4: Downsample [   temp   ]  by 2 (factor of 16 overall) store into [ output ]
*/
void downsample_tiered(float * input, float * output) {
	int x,y,i;
	int x_limit = X_WIDTH, y_limit = Y_WIDTH;;
	int iterations = log_2(BLOCK_WIDTH);
	float * temp_input = input;
	float * temp_output;
	
	for (i = 1; i <= iterations; i++) {
		if(i == iterations) temp_output = output;
		else temp_output = init_matrix(x_limit >> 1, y_limit >> 1);
		
		for (x = 0; x < x_limit; x+=2) {
			for (y = 0; y < y_limit; y+=2) {
				float sum = 0.0f;
				sum += temp_input[CALC_INDEX_MOD(x  , y  , x_limit)];
				sum += temp_input[CALC_INDEX_MOD(x  , y+1, x_limit)];
				sum += temp_input[CALC_INDEX_MOD(x+1, y  , x_limit)];
				sum += temp_input[CALC_INDEX_MOD(x+1, y+1, x_limit)];
				
				temp_output[CALC_INDEX_MOD(x >> 1, y >> 1, x_limit >> 1)] = sum / 4;
			}
		}		
		
		x_limit >>= 1;
		y_limit >>= 1;
		
		if (i > 1) {
			free(temp_input);
		}
		
		temp_input = temp_output;
	}
}

/*
* Function for smoothing a matrix
* 
* Takes every element in the output matrix (excluding border elements)
* and takes an average of its neighbors in all directions
*
*   N N N
*	N E N	where N -> Neighbor
*	N N N		  E -> Current Element
*
*/
void smooth(const float * input, float * output) {
	int i,j;
	
	for (i = 1; i < X_WIDTH - 1; i++) {
		for (j = 1; j < Y_WIDTH - 1; j++) {
			float sum = 0.0f;
			sum+= input[CALC_INDEX(i-1, j-1)];
			sum+= input[CALC_INDEX(i-1, j)];
			sum+= input[CALC_INDEX(i-1, j+1)];
			
			sum+= input[CALC_INDEX(i, j-1)];
			sum+= input[CALC_INDEX(i, j)]; //add ourselves to the average
			sum+= input[CALC_INDEX(i, j+1)];
			
			sum+= input[CALC_INDEX(i+1, j-1)];
			sum+= input[CALC_INDEX(i+1, j)];
			sum+= input[CALC_INDEX(i+1, j+1)];
			
			output[CALC_INDEX(i,j)] = sum / 9;
		}	
	}	
}
/*
* Testing Functions End
***************************************************************************************/
