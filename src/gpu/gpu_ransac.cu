#include <iostream>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#include "gpu_math.cuh"
#include "gpu_ransac.cuh"

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

__global__ void gpu_get_close_points_indices(
    int index1, 
    int index2, 
    const Point3D *points,
    int num_of_points,
    float epsilon,
    int *close_points_indices,
    int *num_of_close_points)
{
    const Point3D p1 = points[ index1 ];
    const Point3D p2 = points[ index2 ];
    const Point3D up = {0, 1, 0};

    float coeffs[4];
    gpu_plane_coeffs(up, p1, p2, coeffs);
    const float tmp_sqrt = sqrtf(      //sqrt(A^2 + B^2 + C^2)
        coeffs[0] * coeffs[0] + 
        coeffs[1] * coeffs[1] + 
        coeffs[2] * coeffs[2]);
    
    int local_close_points = 0;

    for(int i = 0; i < num_of_points; ++i)
    {
        Point3D tmp = points[i];
        if( (tmp.x > 1.2 || tmp.x < -0.2) &&
            (tmp.z > 0.3 || tmp.z < -1.1) &&
            gpu_distance_from_plane(tmp, coeffs, tmp_sqrt) < epsilon)
        {
            close_points_indices[local_close_points] = i;
            ++local_close_points;
        }
    }
    *num_of_close_points = local_close_points;
    //return num_of_close_points;
}

__device__ int gpu_count_points_close_to_points(
    int index1, 
    int index2, 
    const Point3D *points,
    int num_of_points,
    float epsilon)
{
    const Point3D p1 = points[ index1 ];
    const Point3D p2 = points[ index2 ];
    const Point3D up = {0, 1, 0};

    float coeffs[4];
    gpu_plane_coeffs(up, p1, p2, coeffs);
    const float tmp_sqrt = sqrtf(      //sqrt(A^2 + B^2 + C^2)
        coeffs[0] * coeffs[0] + 
        coeffs[1] * coeffs[1] + 
        coeffs[2] * coeffs[2]);
    
    int num_of_close_points = 0;

    for(int i = 0; i < num_of_points; ++i)
    {
        Point3D tmp = points[i];
        if( (tmp.x > 1.2 || tmp.x < -0.2) &&
            (tmp.z > 0.3 || tmp.z < -1.1) &&
            gpu_distance_from_plane(tmp, coeffs, tmp_sqrt) < epsilon)
        {
            ++num_of_close_points;
        }
    }

    return num_of_close_points;
}


__global__ void gpu_count_close_points(
    const Point3D *points,
    const int *randoms,
    const float *valid_points,
    int num_of_points,
    float epsilon,
    int *num_of_close_points)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;

    num_of_close_points[ind] = gpu_count_points_close_to_points(
                                   randoms[2 * ind],
                                   randoms[2 * ind + 1],
                                   points,
                                   num_of_points,
                                   epsilon);    
} 

int get_points_close_to_plane(
    int index,
    int iter_num,
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    std::vector<int> &close_points_indices)
{
    const int num_of_threads = 512;
    //const int num = 1024 * 10;
    const int max_close_points = 500;
    const int num_of_points = points.size();

    Point3D *gpu_points;
    int *gpu_randoms, *gpu_close_points;
    int *gpu_num_of_close_points;
    float *gpu_valid_points;

    cudaMalloc((void **) &gpu_points, num_of_points * sizeof(Point3D));
    cudaMalloc((void **) &gpu_randoms, randoms.size() * sizeof(int));
    cudaMalloc((void **) &gpu_valid_points, num_of_points * sizeof(int));
    cudaMalloc((void **) &gpu_close_points, max_close_points * sizeof(int));
    cudaMalloc((void **) &gpu_num_of_close_points, iter_num * sizeof(int));

    cudaMemcpy(gpu_points, points.data(), num_of_points * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_randoms, randoms.data(), randoms.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    gpu_count_close_points<<<iter_num / num_of_threads, num_of_threads>>>(
        gpu_points, gpu_randoms, gpu_valid_points, 
        num_of_points, epsilon, gpu_num_of_close_points);

    cudaDeviceSynchronize();

    close_points_indices.resize(iter_num);
    cudaMemcpy(close_points_indices.data(), gpu_num_of_close_points, iter_num * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpu_points);
    cudaFree(gpu_randoms);
    cudaFree(gpu_close_points);
    cudaFree(gpu_num_of_close_points);

    return 0;
}