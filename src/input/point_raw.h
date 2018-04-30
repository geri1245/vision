#ifndef UTIL_POINT_RAW_H
#define UTIL_POINT_RAW_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Point3D
{
    float x, y, z;
};

#endif