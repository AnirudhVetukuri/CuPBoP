#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

int main() {
    int numRows = 1024;
    int numCols = 1024;
    size_t bytes = numRows * numCols * sizeof(float);
    float alpha = 1.0f;
    float beta = 1.0f;

    printf("Allocating memory for matrices on host...\n");

    // Allocate memory for matrices on host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    printf("Initializing matrices on host...\n");

    // Initialize matrices on host
    for(int i = 0; i < numRows * numCols; i++) {
        h_A[i] = 1.0f; // Example values
        h_B[i] = 2.0f;
    }

    printf("Allocating memory for matrices on device...\n");

    // Allocate memory for matrices on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    printf("Copying matrices from host to device...\n");

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Create cuBLAS context
    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("Performing matrix addition...\n");

    // Perform matrix addition: C = alpha*A + beta*B
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, numRows, numCols, &alpha, d_A, numRows, &beta, d_B, numRows, d_C, numRows);

    printf("Copying result to host...\n");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Cleaning up...\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cublasDestroy(handle);

    printf("Done.\n");

    return 0;
}