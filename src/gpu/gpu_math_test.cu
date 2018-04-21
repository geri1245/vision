#include "gpu_math.cuh"
#include <iostream>

__global__ void gpu_add_points_test(Point3D a, Point3D b, Point3D *c)
{
    *c = gpu_add_points(a, b);
}

__global__ void gpu_matrix_mul_3x3_test(float *A, float *B, float *result)
{
    gpu_matrix_mul_3x3(A, B, result);
}

__global__ void gpu_mat_vec_mul_test(float *mat, float *vec, float *res)
{
    gpu_mat_vec_mul(mat, vec, res);
    Point3D tmp = gpu_mat_vec_mul(mat, (Point3D){vec[0], vec[1], vec[2]});

    //Here tmp == res

    res[0] = tmp.x;
    res[1] = tmp.y;
    res[2] = tmp.z;
}

__global__ void gpu_cross_vec3_test(float *u, float *v, float *res)
{
    gpu_cross_vec3(u, v, res);
}

__global__ void gpu_plane_coeffs_test(Point3D vec, Point3D p1, Point3D p2, float *coeffs)
{
    gpu_plane_coeffs(vec, p1, p2, coeffs);
}

__global__ void gpu_distance_from_plane_test(Point3D p, float *coeffs, float *dist)
{
    *dist = gpu_distance_from_plane(p, coeffs);
}

__global__ void gpu_dot_point_test(Point3D p1, Point3D p2, float *result)
{
    *result = gpu_dot_point(p1, p2);
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
    
    gpu_cross_vec3_test<<<1, 1>>>(gpu_u, gpu_v, gpu_resu);
    
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

    //Matrix vector multiplication test

    float mat[] = {1, 8, 3, 4, 5, 2, 3, 9, 10};
    float vec[] = {1, 5, 4};

    float *gpu_mat, *gpu_vec, *gpu_res_vec;
    float res_vec[3];

    cudaMalloc((void**)&gpu_mat, sizeof(mat));
    cudaMalloc((void**)&gpu_vec, sizeof(vec));
    cudaMalloc((void**)&gpu_res_vec, sizeof(vec));

    cudaMemcpy(gpu_mat, mat, sizeof(mat), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_vec, vec, sizeof(vec), cudaMemcpyHostToDevice);

    gpu_mat_vec_mul_test<<<1, 1>>>(gpu_mat, gpu_vec, gpu_res_vec);

    //Dot product test
    Point3D p_dot1 = (Point3D){2, 4.7, 10};
    Point3D p_dot2 = (Point3D){50, 4, 1.14};

    float *gpu_dot_result, dot_result;

    cudaMalloc((void**) &gpu_dot_result, sizeof(float));

    gpu_dot_point_test<<<1, 1>>>(p_dot1, p_dot2, gpu_dot_result);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&dist, gpu_dist, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res_vec, gpu_res_vec, sizeof(res_vec), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dot_result, gpu_dot_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_dist);
    cudaFree(gpu_mat);
    cudaFree(gpu_vec);
    cudaFree(gpu_res_vec);
    cudaFree(gpu_dot_result);
    
    std::cout << "\nPoint distance from plane:\n" << dist;

    std::cout << "\nMatrix vector multiplication result:\n";
    for(int i = 0; i < 3; ++i)
    {
        std::cout << res_vec[i] << " ";
    }

    std::cout << "\nDot product result: " << dot_result;

    std::cout << "\n";

    return 0;
}