#include <vector>
#include <iostream>

__global__ void set(bool *values, int *indices, int num)
{
  int ind = threadIdx.x;
    values[ indices[ind] ] = true;
}

int main()
{
  bool *gpu_values;
  int *gpu_indices;

  bool values[50];
  bool res_values[50];

  for(int i = 0; i < 50; ++i)
  {
    values[i] = false;
  }

  int indices[25];

  for(int i = 0; i < 25; ++i)
  {
    indices[i] = 2 * i;
  }

  cudaMalloc(&gpu_values, 50 * sizeof(bool));
  cudaMalloc(&gpu_indices, 50 * sizeof(int));

  cudaMemcpy(gpu_values, values, 50 * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_indices, indices, 25 * sizeof(int), cudaMemcpyHostToDevice);
  
  set<<<1, 40>>>(gpu_values, gpu_indices, 25);

  int indices2[15];
  for(int i = 0; i < 15; ++i)
  {
    indices2[i] = i;
  }

  cudaMemcpy(gpu_indices, indices2, 15 * sizeof(int), cudaMemcpyHostToDevice);
  set<<<1, 40>>>(gpu_values, gpu_indices, 15);
  
  cudaMemcpy(res_values, gpu_values, 50 * sizeof(bool), cudaMemcpyDeviceToHost);
  
  for(int i = 0; i < 50; ++i)
  {
    std::cout << res_values[i] << " ";
  }

  return 0;
}
