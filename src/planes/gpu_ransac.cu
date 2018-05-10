#include <iostream>
#include <vector>
#include <algorithm>

#include <curand.h>
#include <curand_kernel.h>

#include "gpu_math.cuh"
#include "gpu_ransac.cuh"
#include "cpu_math.h"

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

__device__ int gpu_get_close_points_indices(
    int index1,
    int index2, 
    const Point3D *points,
    int num_of_points,
    float epsilon,
    int *close_points_indices,
    const bool *valid_points)
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
        if( valid_points[i] &&
            (tmp.x > 1.2 || tmp.x < -0.2) &&
            (tmp.z > 0.3 || tmp.z < -1.1) &&
            gpu_distance_from_plane(tmp, coeffs, tmp_sqrt) < epsilon)
        {
            //if(num_of_close_points < 150)
            //close_points_indices[num_of_close_points] = 3;
            
            ++num_of_close_points;
        }
    }
    
    return num_of_close_points;
}

__device__ int gpu_count_points_close_to_plane(
    int index1,
    int index2, 
    const Point3D *points,
    int num_of_points,
    float epsilon,
    const bool *valid_points)
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
        if( valid_points[i] &&
            (tmp.x > 1.2 || tmp.x < -0.2) &&
            (tmp.z > 0.3 || tmp.z < -1.1) &&
            tmp.y > -0.55 &&
            gpu_distance_from_plane(tmp, coeffs, tmp_sqrt) < epsilon)
        {
            ++num_of_close_points;
        }
    }

    return num_of_close_points;
}

__global__ void set_to_false(bool *values, int *indices, int num)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if(ind < num)
        values[ indices[ind] ] = false;
}

__global__ void init_to_true(bool *values, int num)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if(ind < num)
        values[ ind ] = true;
}


__global__ void gpu_count_close_points(
    const Point3D *points,
    const int *randoms,
    const bool *valid_points,
    int num_of_points,
    float epsilon,
    int *num_of_close_points)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;

    num_of_close_points[ind] = gpu_count_points_close_to_plane(
                                   randoms[2 * ind],
                                   randoms[2 * ind + 1],
                                   points,
                                   num_of_points,
                                   epsilon,
                                   valid_points);
} 

int get_points_close_to_plane(
    int iter_num,
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    int threshhold,
    std::vector< std::vector<int> > &plane_points)
{
    //************
    //Initialize variables of this frame
    //************
    const int num_of_threads = 512;
    const int num_of_points = points.size();
    const int max_num_of_planes = 5;
    std::vector<int> gpu_results;

    Point3D *gpu_points;
    int *gpu_randoms;
    int *gpu_num_of_close_points;
    int *gpu_indices;
    bool *gpu_valid_points;

    cudaMalloc((void **) &gpu_points, num_of_points * sizeof(Point3D));
    cudaMalloc((void **) &gpu_randoms, randoms.size() * sizeof(int));
    cudaMalloc((void **) &gpu_indices, 512 * sizeof(int));
    cudaMalloc((void **) &gpu_valid_points, num_of_points * sizeof(bool));
    cudaMalloc((void **) &gpu_num_of_close_points, iter_num * sizeof(int));

    cudaMemcpy(gpu_points, points.data(), num_of_points * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_randoms, randoms.data(), randoms.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    int num_of_blocks = (num_of_points + num_of_threads - 1) / num_of_threads;
    init_to_true<<<num_of_blocks, num_of_threads>>>(gpu_valid_points, num_of_points);

    //*****************
    //Start of meaningful work
    //*****************

    plane_points.clear();
    plane_points.reserve(15);
    gpu_results.resize(15);

    for(int i = 0; i < max_num_of_planes; ++i)
    {
        //We count how many points are within epsilon distance from
        //each plane
        gpu_count_close_points<<<iter_num / num_of_threads, num_of_threads>>>(
            gpu_points, gpu_randoms, gpu_valid_points, 
            num_of_points, epsilon, gpu_num_of_close_points);
    
        cudaDeviceSynchronize();

        gpu_results.resize(iter_num);
        cudaMemcpy(gpu_results.data(), 
                   gpu_num_of_close_points, 
                   iter_num * sizeof(int), 
                   cudaMemcpyDeviceToHost);

        std::vector<int>::iterator it = std::max_element(gpu_results.begin(), gpu_results.end());
        if(*it < threshhold)
            break; //Not a big enough plane, we stop

        std::vector<int> tmp;
        plane_points.push_back(tmp); //Can't use move yet
        plane_points[i].resize(*it);

        int num_of_close_points = get_close_points_indices(
            std::distance(gpu_results.begin(), it), points,
            randoms, epsilon,
            plane_points[i]);

        cudaMemcpy(gpu_indices, plane_points[i].data(), 
                   num_of_close_points * sizeof(int), 
                   cudaMemcpyHostToDevice);

        //Set the indices of points of the current plane as not valid
        //so we don't find the exact same plane again
        const int thread_num = num_of_close_points < 256 ? 256 : 512;
        set_to_false<<<1, thread_num>>>(
            gpu_valid_points, gpu_indices, num_of_close_points);
    }


    cudaFree(gpu_points);
    cudaFree(gpu_indices);
    cudaFree(gpu_randoms);
    cudaFree(gpu_valid_points);
    cudaFree(gpu_num_of_close_points);

    return 0;
}