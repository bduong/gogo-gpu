//nvcc -I/ad/eng/support/software/linux/all/x86_64/cuda/cuda_sdk/C/common/inc -I/ad/eng/support/software/linux/all/x86_64/cuda/cuda/include/ -L/ad/eng/support/software/linux/all/x86_64/cuda/cuda/lib64/ -lcuda -lcudart -lm project.cu -o project

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>     
#include <time.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_functions.h>
#include "defines.h"

//#define SM_ARR_LEN		8000
//#define FACTOR                  8     //downsample factor
#define OUT_ARR_LEN		SM_ARR_LEN
//#define OUT_ARR_LEN             SM_ARR_LEN/FACTOR    //downsample array length
#define TILE_WIDTH              8
#define BLOCKS                  OUT_ARR_LEN/TILE_WIDTH
#define TOL			1710e-6


#define GIG 1000000000
#define CPG 2.53           // Cycles per GHz -- Adjust to your computer
/*
* CUDA Kernel Device Code
* Computes the vector addition of A and B into C. The three vectors have the same
* number of elements numElements.
*/
//Smoothing with shared memory
__global__ void smoothing_s(const float* input, float* output)
{
    __shared__ float input_s[TILE_WIDTH][TILE_WIDTH];    

    int bx = blockIdx.x; int by = blockIdx.y; // ID thread
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;

    input_s[tx][ty] = input[Row*SM_ARR_LEN + Col];
    __syncthreads();
    
    if (((tx>0) && (tx < TILE_WIDTH-1))&&((ty>0) && (ty < TILE_WIDTH-1)))
    {
	sum=0.25*(input_s[tx][ty]);  //center value
	sum+=0.125*(input_s[tx][ty-1] + input_s[tx][ty+1]);  //top and bottom
        sum+=0.125*(input_s[tx-1][ty] + input_s[tx-1][ty]);  //left and right
        sum+=0.0625*(input_s[tx-1][ty-1] + input_s[tx+1][ty-1]);  //upper corners
	sum+=0.0625*(input_s[tx-1][ty+1] + input_s[tx+1][ty+1]);  //lower corners
    }

    output[Row*SM_ARR_LEN + Col] = sum/9.0f;
}

//Smoothing blocked    
__global__ void smoothing(const float* input, float* output)
{
    int bx = blockIdx.x; int by = blockIdx.y; // ID thread
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;
    
    if (((Row>0) && (Row < SM_ARR_LEN-1))&&((Col>0) && (Col < SM_ARR_LEN-1)))
    {
	sum=0.25*(input[Row*SM_ARR_LEN + Col]);  //center value
	sum+=0.125*(input[(Row-1)*SM_ARR_LEN+Col] + input[(Row+1)*SM_ARR_LEN+Col]);  //top and bottom
        sum+=0.125*(input[Row*SM_ARR_LEN + (Col-1)] + input[Row*SM_ARR_LEN + (Col+1)]);  //left and right
        sum+=0.0625*(input[(Row-1)*SM_ARR_LEN+(Col-1)] + input[(Row-1)*SM_ARR_LEN+(Col+1)]);  //upper corners
	sum+=0.0625*(input[(Row+1)*SM_ARR_LEN+(Col-1)] + input[(Row+1)*SM_ARR_LEN+(Col+1)]);  //lower corners
    }

    output[Row*SM_ARR_LEN + Col] = sum/9.0f;
}

/*
//Smoothing unblocked    
__global__ void smoothing(const float* input, float* output)
{

    int tx = threadIdx.x; int ty = threadIdx.y;
    float sum = 0.0f;
    
    if (((tx>0) && (tx < SM_ARR_LEN-1))&&((ty>0) && (ty < SM_ARR_LEN-1)))
    {
	sum=0.25*(input[ty * SM_ARR_LEN + tx]);  //center value
	sum+=0.125*(input[(ty-1)*SM_ARR_LEN+tx] + input[(ty+1)*SM_ARR_LEN+tx]);  //top and bottom
        sum+=0.125*(input[ty*SM_ARR_LEN + (tx-1)] + input[ty*SM_ARR_LEN + (tx+1)]);  //left and right
        sum+=0.0625*(input[(ty-1)*SM_ARR_LEN+(tx-1)] + input[(ty-1)*SM_ARR_LEN+(tx+1)]);  //upper corners
	sum+=0.0625*(input[(ty+1)*SM_ARR_LEN+(ty-1)] + input[(ty+1)*SM_ARR_LEN+(ty+1)]);  //lower corners
    }

    output[tx * SM_ARR_LEN + ty] = sum;
}
*/
/*
//Downsample with shared memory
__global__ void downsample_s(const float* input, float* output, int Width)
{
    __shared__ float input_s[TILE_WIDTH*FACTOR][TILE_WIDTH*FACTOR];

    int bx = blockIdx.x; int by = blockIdx.y; // ID thread
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0; // REGISTER!

    for (int j = 0; j<FACTOR; ++j){
	for (int k = 0; k < FACTOR; ++k)
            input_s[tx*FACTOR+j][ty*FACTOR+k] = input[(Row*FACTOR+j)*Width + Col*FACTOR + k];
        __syncthreads();
    }

    for (int x = 0; x<FACTOR; ++x)
	for (int y = 0; y < FACTOR; ++y)
            Pvalue+=input_s[x][y];
    

    output[Row*OUT_ARR_LEN+Col] = Pvalue/(FACTOR*FACTOR);
}
*/
/*
__global__ void downsample(const float* input, float* output, int Width)
{
    int bx = blockIdx.x; int by = blockIdx.y; // ID thread
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0; // REGISTER!

    for (int j = 0; j<FACTOR; ++j)
	for (int k = 0; k < FACTOR; ++k)
		Pvalue += input[(Row*FACTOR+j)*Width + Col*FACTOR + k];

    output[Row*OUT_ARR_LEN+Col] = Pvalue/(FACTOR*FACTOR);
}
*/
/**
* Host Main routine
*
*/

int main(void)
{
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2, time3, time4;
    struct timespec time_stamp[2];
    float difference;

    StopWatchInterface *kernelTime = 0;

    sdkCreateTimer(&kernelTime);
    sdkResetTimer(&kernelTime);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    // Print the vector length to be used and compute it's size
    int length = SM_ARR_LEN;
    int numElements = SM_ARR_LEN*SM_ARR_LEN;
    int numElementsout = OUT_ARR_LEN*OUT_ARR_LEN;
    size_t size = numElements * sizeof(float);
    size_t size_out = numElementsout * sizeof(float);  //output matrix
//    printf("Downsample of %d elements \n",numElements);
//    printf("Output of %d elements \n",numElementsout);

    // Allocate HOST MEMORY
    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size_out);  

    // Initial the host input vectors
    srand(1);
    for(int i =0; i < numElements; i++)
    {
	h_A[i] = rand()/ (float)RAND_MAX;
    }
    for(int j =0; j < numElementsout; j++)
    {
	h_B[j] = 0.0f;
    }

      sdkStartTimer(&kernelTime);  

//    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time3); //get start time for cuda
    // Allocate DEVICE vectors
    float *d_A = NULL;
    float *d_B = NULL;

    err = cudaMalloc((void**)&d_A,size);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_B,size_out);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    // Copy Matrices to DEVICE
  //  printf("Copy input data from the host memory to the CUDA device \n");   
    err = cudaMemcpy(d_A, h_A, size , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Matrix A from hos to device (error code %s)! \n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_B, h_B, size_out , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Matrix B from hos to device (error code %s)! \n", cudaGetErrorString(err));
    }

//    printf("CUDA kernel launch with %d blocks of %d threads \n", BLOCKS, TILE_WIDTH*TILE_WIDTH);

    //This is setup for all functions except smoothing without blocking
    dim3 blocksPerGrid(BLOCKS,BLOCKS,1);
    dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH,1);

    //This is setup for smoothing without blocking
//    dim3 dimGrid(1,1,1);
//    dim3 dimBlock(length, length,1);

//    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
//    sdkStartTimer(&kernelTime);
    for (int j=0; j<200; j++)
//    downsample <<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, length);
//    downsample_s <<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, length);
//    smoothing <<< blocksPerGrid, threadsPerBlock>>>(d_A, d_B);
    smoothing_s <<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B);
    cudaDeviceSynchronize();
//    sdkStopTimer(&kernelTime);
//    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);


    err = cudaGetLastError();

    if( err != cudaSuccess)
    {
	fprintf(stderr, "Failed to launch kernel (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    // Copy the device result vector in device memory to the host result vector
    // in host memory
//    printf("Copy output data from CUDA device to the host memory \n");
    err = cudaMemcpy(h_B,d_B,size_out,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }    

    // Free device global memory
    err = cudaFree(d_A);
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to free device matrix A (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to free device matrix B (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to deinitialize the device! (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
//    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time4);
//    time_stamp[0] = diff(time3,time4);

    cudaDeviceSynchronize();
    sdkStopTimer(&kernelTime);

//    for(int i=0; i < numElementsout; i++)
//    {
//        difference = abs(h_B[i]-h_B[i]);
//        difference = abs(h_B[i]-1.0);
//	if( difference > TOL)
//	{
//	    fprintf(stderr, "Result verification failed at element %d\n",i);
//            printf("GPU: %f     CPU: %f     difference: %f\n", h_A[i], h_B[i], difference);
//            printf("GPU: %f     CPU: %f     index: %d\n", h_A[i], h_B[i], i);
//            printf("difference: %f\n", difference);
//	    exit(EXIT_FAILURE);
//	}
//    }

    //PRINT TIME FOR CUDA
//    printf("cuda time: %ld\n", (long int)((double)(CPG)*(double)
//		 (GIG * time_stamp[0].tv_sec + time_stamp[0].tv_nsec)));
//    printf ("Time for the kernel: %f ms\n", sdkGetTimerValue(&kernelTime));
    printf (" %f \n", sdkGetTimerValue(&kernelTime) /1000);
    
// free hosts memory
    free(h_A);
    free(h_B);

//    printf("DONE \n");
    return 0;
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
