//nvcc -I/ad/eng/support/software/linux/all/x86_64/cuda/cuda_sdk/C/common/inc -I/ad/eng/support/software/linux/all/x86_64/cuda/cuda/include/ -L/ad/eng/support/software/linux/all/x86_64/cuda/cuda/lib64/ -lcuda -lcudart -lm project.cu -o project

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>     
#include <time.h>

#define SM_ARR_LEN		100
#define FACTOR                  2     //downsample factor
#define OUT_ARR_LEN             SM_ARR_LEN/FACTOR
#define TILE_WIDTH              10
#define BLOCKS                  OUT_ARR_LEN/TILE_WIDTH
#define TOL			1710e-6


#define GIG 1000000000
#define CPG 2.53           // Cycles per GHz -- Adjust to your computer
/*
* CUDA Kernel Device Code
* Computes the vector addition of A and B into C. The three vectors have the same
* number of elements numElements.
*/

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

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    // Print the vector length to be used and compute it's size
    int factor = FACTOR;
    int length = SM_ARR_LEN;
    int numElements = SM_ARR_LEN*SM_ARR_LEN;
    int numElementsout = OUT_ARR_LEN*OUT_ARR_LEN;
    size_t size = numElements * sizeof(float);
    size_t size_out = numElementsout * sizeof(float);  //output matrix
    printf("Downsample of %d elements \n",numElements);
    printf("Output of %d elements \n",numElementsout);

    // Allocate HOST MEMORY
    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size_out);  

    // Initial the host input vectors
    srand(1);
    for(int i =0; i < numElements; i++)
    {
	h_A[i] = rand()/ (float)RAND_MAX;
//	h_A[i] = 1.0f;
    }
    for(int j =0; j < numElementsout; j++)
    {
	h_B[j] = 0.0f;
    }

    
//    if (0)   //don't do CUDA
//   {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time3); //get start time for cuda
    // Allocat DEVICE vectors
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
    printf("Copy input data from the host memory to the CUDA device \n");   
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

    printf("CUDA kernel launch with %d blocks of %d threads \n", BLOCKS, TILE_WIDTH*TILE_WIDTH);

    dim3 blocksPerGrid(BLOCKS,BLOCKS,1);
    dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH,1);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    downsample <<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, length);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);


    err = cudaGetLastError();

    if( err != cudaSuccess)
    {
	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    // Copy the device result vector in device memory to the host result vector
    // in host memory
    printf("Copy output data from CUDA device to the host memory \n");
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
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time4);
    time_stamp[0] = diff(time3,time4);
//    } //-----------don't do gpu

/*    
    if (0) //don't do cpu
    {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    bijk(h_A, h_B, h_D, TILE_WIDTH);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    
    time_stamp[1] = diff(time1,time2);
    } //----------don't do cpu
*/

    for(int i=0; i < numElementsout; i++)
    {
//        difference = abs(h_B[i]-h_B[i]);
//        difference = abs(h_B[i]-1.0);
//	if( difference > TOL)
//	{
//	    fprintf(stderr, "Result verification failed at element %d\n",i);
//            printf("GPU: %f     CPU: %f     difference: %f\n", h_A[i], h_B[i], difference);
            printf("GPU: %f     CPU: %f     index: %d\n", h_A[i], h_B[i], i);
//            printf("difference: %f\n", difference);
//	    exit(EXIT_FAILURE);
//	}
    }

    printf("cuda time: %ld\n", (long int)((double)(CPG)*(double)
		 (GIG * time_stamp[0].tv_sec + time_stamp[0].tv_nsec)));
    printf("\n");
    printf("cpu time: %ld\n", (long int)((double)(CPG)*(double)
		 (GIG * time_stamp[1].tv_sec + time_stamp[1].tv_nsec)));
    printf("\n");
    
// free hosts memory
    free(h_A);
    free(h_B);

    printf("DONE \n");
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
