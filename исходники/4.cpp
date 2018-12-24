#define N 150000000
#define M 1024 // THREADS_PER_BLOCK
#include<stdio.h>
#include<stdlib.h>
__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N) c[index] = a[index] + b[index];
}
void random_ints(int *a, int n)
{
	srand(time(NULL));
	for(int i=0;i<n;i++)
	{
		a[i]=rand();
	}
}
void check_results(int *a,int *b,int *c,int n)
{
	for(int i=0;i<n;i++)
	{
		if(!(a[i]+b[i]==c[i]))
		{ 
			printf("Error");
			break;
		}
	}
}
int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = N * sizeof(int);
	printf("N=150000000 \n");
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)calloc(N, sizeof(int));
	// Copy inputs to device

	cudaEvent_t start1, stop1;
	float time = 0;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1 , 0);


	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	cudaEventRecord( stop1,0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time, start1, stop1);
	printf("Elapsed time : %.2f ms\n", time);
	cudaEventDestroy(start1);
	cudaEventDestroy (stop1);

	
	cudaEvent_t start, stop;
	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start , 0);
	
	add<<<(N + M - 1) / M, M>>>(d_a, d_b, d_c); // Launch add() kernel on GPU with N blocks
	
	cudaEventRecord( stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed time : %.2f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy (stop);



	cudaEvent_t start2, stop2;
	time = 0;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2 , 0);

	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); // Copy result back to host

	cudaEventRecord( stop2,0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time, start2, stop2);
	printf("Elapsed time : %.2f ms\n", time);
	cudaEventDestroy(start2);
	cudaEventDestroy (stop2);


	check_results(a,b,c, N);
	
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	
	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}