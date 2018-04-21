#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

__global__ void gpu_rand_ints(int *rand_nums, int max)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init(ind, 0, 0, &state);
    rand_nums[ind] = curand(&state) % max;
}

__device__ int gpu_rand_int(int seed, int max)
{
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    return curand(&state) % max;
}

__global__ void count(const Point3D *points, int *num_of_close_points, int *random_point_indices, int num_of_points)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;

    
} 

int main()
{
    //Random number generation
    //************************
    const int num_of_threads = 512;
    const int num = 4096 * 10;

    int *gpu_rand, max = 30000;
    int random_numbers[num];

    cudaMalloc((void **) &gpu_rand, sizeof(random_numbers));
    
    //gpu_rand_ints<<<num / num_of_threads, num_of_threads>>>(gpu_rand, max);

    cudaDeviceSynchronize();

    cudaMemcpy(random_numbers, gpu_rand, num * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpu_rand);

    //Allocation test
    //***************

    int *numbers, capacity = 400;
    int *gpu_numbers, *gpu_capacity;

    cudaMalloc((void **) &gpu_numbers, capacity * sizeof(float));
    cudaMalloc((void **) &gpu_capacity, sizeof(float));

    cudaMemcpy(gpu_capacity, &capacity, sizeof(int), cudaMemcpyHostToDevice);
    allocate<<<1, 1>>>(gpu_numbers, gpu_capacity);

    cudaDeviceSynchronize();

    cudaMemcpy(&capacity, gpu_capacity, sizeof(int), cudaMemcpyDeviceToHost);

    numbers = new int[capacity]; 

    cudaMemcpy(numbers, gpu_numbers, capacity * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < capacity; ++i)
    {
        std::cout << numbers[i] << " ";
    }
    cudaFree(gpu_numbers);
    cudaFree(gpu_capacity);
    delete[] numbers;
}