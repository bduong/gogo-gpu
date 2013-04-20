#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GIG 1000000000
#define CPG 2.52           // Cycles per GHz -- Adjust to your computer

#define X_WIDTH 2000
#define Y_WIDTH 2000
#define FRAND_MIN -10
#define FRAND_MAX 10
#define BLOCK_WIDTH 8
#define OUTPUT_X_WIDTH X_WIDTH/BLOCK_WIDTH
#define OUTPUT_Y_WIDTH Y_WIDTH/BLOCK_WIDTH


float * init_matrix(int x_width, int y_width);
struct timespec diff(struct timespec, struct timespec);
double fRand(double fmin, double fmax);
void seed_matrix(float * matrix, int size);
float * downsample(const float * matrix, float * output);
int clock_gettime(clockid_t clk_id, struct timespec *tp);

main(int argc, char * argv[]) {

   float * matrix, *output;
   struct timespec start, end, difference;

   matrix = init_matrix(X_WIDTH, Y_WIDTH);
   srand(1);
   seed_matrix(matrix, X_WIDTH*Y_WIDTH);

   output = init_matrix(OUTPUT_X_WIDTH, OUTPUT_Y_WIDTH);

   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
   downsample(matrix, output);   
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
   difference = diff(start, end);

   printf("Time to downsample %d x %d: %ld ns\n", X_WIDTH, Y_WIDTH, (long int) (GIG * difference.tv_sec + difference.tv_nsec));
   free(matrix);
   free(output);

}

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

float * downsample(const float * matrix, float * output) {
   int x,y, xx, yy;
   
   for (x = 0; x < X_WIDTH; x+= BLOCK_WIDTH) {
       for (y = 0; y < Y_WIDTH; y += BLOCK_WIDTH) {
	   float sum = 0.0f;
 	   for (xx = x; xx < x+BLOCK_WIDTH; xx++) {
		for (yy = y; yy < y+BLOCK_WIDTH; yy++) {
		   sum += matrix[xx *X_WIDTH + yy];
		}
	   }

	   output[(x/BLOCK_WIDTH) * OUTPUT_X_WIDTH + (y/BLOCK_WIDTH)] = sum / (BLOCK_WIDTH*BLOCK_WIDTH);
	}
   }

}

