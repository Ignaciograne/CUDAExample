
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h> // Para usar atoi

int *a, *b;  // host data
int *c, *c2;  // results on GPU and CPU

void cudaCheckError() {
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) {
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
        exit(0);
    }
}

__global__
void vecAdd(int *A,int *B,int *C,int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void vecAdd_h(int *A1, int *B1, int *C1, int N) {
    for(int i=0; i<N; i++)
        C1[i] = A1[i] + B1[i];
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <vector_size> <block_size>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);  // Tamaño del vector
    int BLOCK_SIZE = atoi(argv[2]);  // Número de hilos por bloque

    int size = N * sizeof(int);
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    cudaMallocManaged(&c2, size);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    clock_t start_d=clock();
    vecAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, N);
    cudaDeviceSynchronize();
    clock_t end_d = clock();

    clock_t start_h=clock();
    vecAdd_h(a, b, c2, N);
    clock_t end_h = clock();

    printf("GPU time: %f seconds\n", (double)(end_d - start_d) / CLOCKS_PER_SEC);
    printf("CPU time: %f seconds\n", (double)(end_h - start_h) / CLOCKS_PER_SEC);

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(c2);

    return 0;
}
