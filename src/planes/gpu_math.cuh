#ifndef GPU_GPU_MATH_CUH
#define GPU_GPU_MATH_CUH

#include "../input/point_raw.h"

//Cross product of float vectors
__device__ void gpu_cross_vec3(const float *u, const float *v, float *res);
//Dot product for vectors represented by Point3D
__device__ Point3D gpu_cross_point(Point3D u, Point3D v);

//Dot product of float vectors
__device__ float gpu_dot_vec3(float *u, float *v);
//Dot product of Point3Ds
__device__ float gpu_dot_point(Point3D u, Point3D v);

__device__ void gpu_matrix_mul_3x3(const float *A, const float *B, float *result);

__device__ void gpu_mat_vec_mul(const float *mat, const float *vec, float *result);
__device__ Point3D gpu_mat_vec_mul(const float *mat, Point3D vec);

__device__ void normalize(Point3D *p);
__device__ Point3D gpu_add_points(Point3D a, Point3D b);
__device__ Point3D gpu_sub_points(Point3D a, Point3D b);

__device__ float gpu_distance_from_plane(Point3D p, float *coeffs);
__device__ float gpu_distance_from_plane(Point3D p, float *coeffs, float sqrt);

//Calculates coefficients for Ax + By + Cz + D = 0 
//From a direction vector of the plane and 2 points on the plane
__device__ void gpu_plane_coeffs(Point3D vec, Point3D p1, Point3D p2, float *coeffs);

#endif