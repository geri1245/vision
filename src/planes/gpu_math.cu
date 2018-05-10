#include <iostream>
#include <stdio.h>
#include <cmath>

#include "gpu_math.cuh"

//Cross product for float vector
__device__ void gpu_cross_vec3(const float *u, const float *v, float *res)
{
    res[0] = u[1] * v[2] - u[2] * v[1];
    res[1] = u[2] * v[0] - u[0] * v[2];
    res[2] = u[0] * v[1] - u[1] * v[0];
}

//Cross product for vectors represented by Point3D
__device__ Point3D gpu_cross_point(Point3D u, Point3D v)
{
    return (Point3D){
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    };
}

__device__ float gpu_dot_vec3(const float *u, const float *v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__device__ float gpu_dot_point(Point3D u, Point3D v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__device__ void gpu_matrix_mul_3x3(const float *A, const float *B, float *result)
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

__device__ void gpu_mat_vec_mul(const float *mat, const float *vec, float *result)
{
    result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
    result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
    result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__device__ Point3D gpu_mat_vec_mul(const float *mat, Point3D vec)
{
    return (Point3D) {
        mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z,
        mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z,
        mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z
    };
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

//sqrt(A^2 + B^2 + C^2) is the last parameter, we don't calculate it every time
__device__ float gpu_distance_from_plane(Point3D p, float *coeffs, float sqrt)
{
    //d = abs( A*x1 + B*y1 + C*z1 ) / sqrt(A^2 + B^2 + C^2) 
    return 
        fabs(coeffs[0] * p.x + coeffs[1] * p.y + coeffs[2] * p.z + coeffs[3]) /
        sqrt;    
}

//Calculates coefficients for Ax + By + Cz + D = 0 
//From a direction vector of the plane and 2 points on the plane
__device__ void gpu_plane_coeffs(Point3D vec, Point3D p1, Point3D p2, float *coeffs)
{
    Point3D vec2 = gpu_sub_points(p1, p2); //Calculate another direction vector
    
    //Calculating the normal vector of the plane
    Point3D normal = gpu_cross_point(vec, vec2);

    coeffs[0] = normal.x;
    coeffs[1] = normal.y;
    coeffs[2] = normal.z;
    coeffs[3] = -normal.x * p1.x - normal.y * p1.y - normal.z * p1.z;
}