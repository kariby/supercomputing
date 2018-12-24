#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<ctime>
#include <stdio.h>
using namespace std;


#define N 100000
#define M 1024 // THREADS_PER_BLOCK
#define streamCount 32

__global__ void add(int *a, int *b, int *c) //выполняется на GPU, вызывается с CPU
{
	int index = threadIdx.x + blockIdx.x * blockDim.x; // 1 - индекс текущей нити в вычислении на GPU, 2 - индекс текущего блока в вычислении на GPU, 3 - размерность блока
	if (index < N) c[index] = a[index] + b[index];
}



//функция заполнения массива случайными числами
void random_ints(int* arr, int n) {
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % 100;
	}
}

//проверка результатов
void checkResults(int* a, int* b, int* c) {
	
	for (int i = 0; i < N; i++) {
		if (a[i] + b[i] != c[i]) {
			cout << "Wrong!" << endl;//
			//return;
		}
	}
	//cout << "Right" << endl;
}

int main(void) {

	srand(time(NULL));
	int *a, *b, *c; //выделяем память под вектора
	int *d_a, *d_b, *d_c; //указатели на память видеокарте
	int size = N*sizeof(int);
		
	cudaMalloc((void**)&d_a,size); //выделяем память для векторов на видеокарте
	cudaMalloc((void**)&d_b,size);
	cudaMalloc((void**)&d_c,size);
	
	cudaMallocHost((void**)&a, size); //страницы вирт. памяти 100% окажутся в опер. памяти
	cudaMallocHost((void**)&b, size);
	cudaMallocHost((void**)&c, size);
	
random_ints(a, N);
	random_ints(b, N);

	cudaStream_t* streams = new cudaStream_t[streamCount]; // поток, в котором будет произведен вызов
	for(int i=0; i<streamCount; ++i)
		cudaStreamCreate(&streams[i]);

	cudaEvent_t start, stop;
	float time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	int width = N / streamCount;
	for(int i = 0; i < streamCount; ++i) 
	{
		int offset = i*width;
		cudaMemcpyAsync(&d_a[offset], &a[offset], size/streamCount, cudaMemcpyDefault, streams[i]);
		cudaMemcpyAsync(&d_b[offset], &b[offset], size/streamCount, cudaMemcpyDefault, streams[i]);
		add <<<(N / streamCount + M - 1) / M, M,0,streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset]);
		cudaMemcpyAsync(&c[offset], &d_c[offset], size/streamCount, cudaMemcpyDefault, streams[i]);
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed time : %f\n", time);	
	
	cout << endl;
	checkResults(a, b, c);
	
	for(int i = 0; i < streamCount; ++i)
		cudaStreamDestroy(streams[i]);
	cudaFreeHost(a);
	cudaFree(d_a);
	cudaFreeHost(b);
	cudaFree(d_b);
	cudaFreeHost(c);
	cudaFree(d_c);

	//delete a; delete b; delete c;
	
	return 0;
}