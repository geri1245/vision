#include <iostream>
#include <stdio.h>
#include <cmath>
#include "../util/point_raw.h"

//Cross product for float vector
__device__ void gpu_cross_3_vec(float *u, float *v, float *res)
{
    res[0] = u[1] * v[2] - u[2] * v[1];
    res[1] = u[2] * v[0] - u[0] * v[2];
    res[2] = u[0] * v[1] - u[1] * v[0];
}

//Cross product for vectors represented by Point3D
__device__ void gpu_cross_3_point(Point3D u, Point3D v, Point3D *res)
{
    res->x = u.y * v.z - u.z * v.y;
    res->y = u.z * v.x - u.x * v.z;
    res->z = u.x * v.y - u.y * v.x;
}

__device__ void gpu_matrix_mul_3x3(float *A, float *B, float *result)
{
    for(int j = 0; j < 3; ++j)
    {
        for(int i = 0; i < 3; ++i)
        {
            //A[i, j] = 3 * i + j;
            result[3 * i + j] = 
                A[3 * i]     * B[j]     + 
                A[3 * i + 1] * B[3 + j] + 
                A[3 * i + 2] * B[6 + j];
        }
    }
}

__device__ void normalize(Point3D *p)
{
    Point3D lp = *p;
    
    float sq = sqrtf(lp.x * lp.x + lp.y * lp.y + lp.z * lp.z);
    
    lp.x = lp.x / sq;
    lp.y = lp.y / sq;
    lp.z = lp.z / sq;

    *p = lp;
}

__device__ Point3D gpu_add_points(Point3D a, Point3D b)
{
    return (Point3D) {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ Point3D gpu_sub_points(Point3D a, Point3D b)
{
    Point3D d = (Point3D) {a.x - b.x, a.y - b.y, a.z - b.z};
    return d;
}

__device__ float gpu_distance_from_plane(Point3D p, float *coeffs)
{
    //d = abs( A*x1 + B*y1 + C*z1 ) / sqrt(A^2 + B^2 + C^2) 
    return 
        fabs(coeffs[0] * p.x + coeffs[1] * p.y + coeffs[2] * p.z + coeffs[3]) /
        sqrtf(coeffs[0] * coeffs[0] + coeffs[1] * coeffs[1] + coeffs[2] * coeffs[2]);    
} 

//Calculates coefficients for Ax + By + Cz + D = 0 
//From a direction vector of the plane and 2 points on the plane
__device__ void gpu_plane_coeffs(Point3D vec, Point3D p1, Point3D p2, float *coeffs)
{
    Point3D vec2 = gpu_sub_points(p1, p2); //Calculate another direction vector
    
    Point3D normal;
    gpu_cross_3_point(vec, vec2, &normal); //Now we have the normal vector of the plane
    coeffs[0] = normal.x;
    coeffs[1] = normal.y;
    coeffs[2] = normal.z;
    coeffs[3] = -normal.x * p1.x - normal.y * p1.y - normal.z * p1.z;
}

__global__ void gpu_add_points_test(Point3D a, Point3D b, Point3D *c)
{
    *c = gpu_add_points(a, b);
}

__global__ void gpu_matrix_mul_3x3_test(float *A, float *B, float *result)
{
    gpu_matrix_mul_3x3(A, B, result);
}

__global__ void gpu_cross_3_vec_test(float *u, float *v, float *res)
{
    gpu_cross_3_vec(u, v, res);
}

__global__ void gpu_plane_coeffs_test(Point3D vec, Point3D p1, Point3D p2, float *coeffs)
{
    gpu_plane_coeffs(vec, p1, p2, coeffs);
}

__global__ void gpu_distance_from_plane_test(Point3D p, float *coeffs, float *dist)
{
    *dist = gpu_distance_from_plane(p, coeffs);
}

int main()
{
    //Matrix multiplication test
    float ma[] = {4.5, 2, 3, 1, 5, 7, 2, 1, 4.5};
    float mb[] = {2, 4, 5, 4, 2, 2, 6, 1, 5};

    float *gpu_ma, *gpu_mb, *gpu_resm;
    float resm[9];

    cudaMalloc((void**)&gpu_ma, sizeof(ma));
    cudaMalloc((void**)&gpu_mb, sizeof(mb));
    cudaMalloc((void**)&gpu_resm, sizeof(mb));

    cudaMemcpy(gpu_ma, ma, sizeof(ma), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mb, mb, sizeof(mb), cudaMemcpyHostToDevice);

    gpu_matrix_mul_3x3_test<<<1, 1>>>(gpu_ma, gpu_mb, gpu_resm);

    //Adding points test
    Point3D a, b, res, *result;
    a.x = 5; a.y = 7; a.z = 10;
    b.x = 15; b.y = 14; b.z = 4;

    cudaMalloc((void **)&result, sizeof(Point3D));

    gpu_add_points_test<<<1, 1>>>(a, b, result);

    //Cross product test
    float u[] = {1, 0, sqrtf(3)};
    float v[] = {1, sqrtf(3), 0};
    float resu[3];

    float *gpu_u, *gpu_v, *gpu_resu;

    cudaMalloc((void**)&gpu_u, sizeof(u));
    cudaMalloc((void**)&gpu_v, sizeof(v));
    cudaMalloc((void**)&gpu_resu, sizeof(v));

    cudaMemcpy(gpu_u, u, sizeof(u), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v, v, sizeof(v), cudaMemcpyHostToDevice);
    
    gpu_cross_3_vec_test<<<1, 1>>>(gpu_u, gpu_v, gpu_resu);
    
    //Plane coefficients test
    Point3D direction = (Point3D) {5, 2, -3};
    Point3D p1 = (Point3D) {1, -1, 3}, p2 = (Point3D) {4, 1, -2};
    float *gpu_coeffs;
    float coeffs[4];

    cudaMalloc((void **)&gpu_coeffs, sizeof(coeffs));

    gpu_plane_coeffs_test<<<1, 1>>>(direction, p1, p2, gpu_coeffs);

    cudaDeviceSynchronize();

    cudaMemcpy(&res, result, sizeof(Point3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(resm, gpu_resm, sizeof(resm), cudaMemcpyDeviceToHost);
    cudaMemcpy(resu, gpu_resu, sizeof(resu), cudaMemcpyDeviceToHost);
    cudaMemcpy(coeffs, gpu_coeffs, sizeof(coeffs), cudaMemcpyDeviceToHost);

    cudaFree(result);
    cudaFree(gpu_ma);
    cudaFree(gpu_mb);
    cudaFree(gpu_resm);
    cudaFree(gpu_u);
    cudaFree(gpu_v);
    cudaFree(gpu_resu);

    std::cout << "Cross product result:\n";
    for(int i = 0; i < 3; ++i)
    {
        std::cout << resu[i] << " ";
    }

    std::cout << "\nPlane coeffs: \n";
    for(int i = 0; i < 4; ++i)
    {
        std::cout << coeffs[i] << " ";
    }

    std::cout << "\nMatrix multiplication result:\n";
    for(int i = 0; i < 9; ++i)
    {
        std::cout << resm[i] << " ";
    }

    //*****************************
    //Second part of test functions
    //*****************************

    //Point distance from plane test
    Point3D point = (Point3D){2, 3, 1};
    float coeffs_distance[4] = {1, -2, 3, -5};
    float *gpu_dist, *gpu_coeffs_distance, dist;

    cudaMalloc((void **)&gpu_dist, sizeof(float));
    cudaMalloc((void **)&gpu_coeffs_distance, sizeof(coeffs_distance));

    cudaMemcpy(gpu_coeffs_distance, coeffs_distance, sizeof(coeffs_distance), cudaMemcpyHostToDevice);

    gpu_distance_from_plane_test<<<1, 1>>>(point, gpu_coeffs_distance, gpu_dist);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&dist, gpu_dist, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_dist);
    
    std::cout << "\nPoint distance from plane:\n" << dist << "\n";

    return 0;
}