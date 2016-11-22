#include <cuda.h>
#include <stdio.h>

// Multiplicacion de Fila - Matriz

__global__ void multMatCUDA(double *d_a, double *d_b, double *d_c, int NumberRows_A, int NumberCols_A, int NumberCols_B)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < NumberRows_A && col < NumberCols_B)
  {
    double result = 0;
    for (int k = 0; k < NumberCols_A; k++)
    {
      result += d_a[row * NumberCols_A + k] * d_b[k * NumberCols_B + col];
    }
    d_c[row * NumberCols_B + col] = result;
  }
}

void Mult_Matrix_Cuda(double *h_a, double *h_b, double *h_c, int NumberRows_A, int NumberCols_A, int NumberCols_B)
{
  int blocksize = 32;
  double *d_a, *d_b, *d_c;

  // Asign memory in the device
  cudaMalloc(&d_a, sizeof(double) * NumberRows_A * NumberCols_A);
  cudaMalloc(&d_b, sizeof(double) * NumberCols_A * NumberCols_B);
  cudaMalloc(&d_c, sizeof(double) * NumberRows_A * NumberCols_B);

  cudaMemcpy(d_a, h_a, NumberRows_A * NumberCols_A * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NumberCols_A * NumberCols_B * sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimBlock(blocksize, blocksize, 1);
  dim3 dimGrid((NumberCols_B / blocksize) + 1, (NumberRows_A / blocksize) + 1);

  multMatCUDA<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, NumberRows_A, NumberCols_A, NumberCols_B);
  cudaMemcpy(h_c, d_c, NumberRows_A * NumberCols_B * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}