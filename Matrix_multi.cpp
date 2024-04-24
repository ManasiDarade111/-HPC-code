#include <iostream>
#include <cstdlib>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    const int N = 4;  // Size of matrices
    const int blockSize = 2;  // Block size

    // Allocate memory on host for matrices
    int *h_a = new int[N * N];
    int *h_b = new int[N * N];
    int *h_c = new int[N * N];

    // Initialize input matrices on host
    for (int i = 0; i < N * N; i++) {
        h_a[i] = rand() % 10;  // Random values for matrix A
        h_b[i] = rand() % 10;  // Random values for matrix B
    }

    // Allocate memory on device for matrices
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    dim3 block(blockSize, blockSize);

    // Launch CUDA kernel for matrix multiplication
    matrixMultiply<<<grid, block>>>(d_a, d_b, d_c, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result matrix
    std::cout << "Result Matrix:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
