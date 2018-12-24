#define N 100000
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
	printf("Correct \n");
}
int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	
	cudaMallocManaged(&a, N);
	cudaMallocManaged(&b, N);
	cudaMallocManaged(&c, N);
	
	cudaEvent_t start, stop;
	float time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start , 0);
	
	add<<<(N + M - 1) / M, M>>>(a, b, c); // Launch add() kernel on GPU with N blocks
	
	cudaEventRecord( stop,0);
	cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed time : %.2f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy (stop);
	
	check_results(a,b,c, N);
	
	
	// Cleanup
	cudaFree(a); cudaFree(b); cudaFree(c);
	return 0;
}