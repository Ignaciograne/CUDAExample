#include <stdio.h>

#define TILE_WIDTH 16 // Block size for the tile

//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0); 
 }
}

// CUDA kernel for matrix multiplication
__global__
void matrixMulKernel(float *A, float *B, float *C, int M, int K, int N) {
    // Compute the row and column index for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we're inside the matrix bounds
    if (row < M && col < N) {
        float value = 0.0f;
        // Compute the dot product for the C[row][col] element
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        // Store the result
        C[row * N + col] = value;
    }
}

int main() {
    // Matrix dimensions
    int M = 4; // A and C height
    int K = 4; // A width and B height
    int N = 4; // B width and C width
    
    // Allocate host matrices A, B, and C
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_C2 = (float*)malloc(M * N * sizeof(float));

    // Initialize the host matrices with values (for testing)
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    // Allocate device matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);  // 16x16 threads per block
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);  // Enough blocks to cover the entire matrix

    clock_t start_d=clock();
    printf("Doing GPU Matrix Multiplication\n");
    // Launch the matrix multiplication kernel
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaCheckError();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();

    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;

    // Copy the result matrix back to the host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Resulting matrix = %d x %d \t GPU time = %f \n", M, N, time_d);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
