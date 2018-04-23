#ifndef GPU_CPU_MATH_H
#define GPU_CPU_MATH_H

#include <vector>
#include "../util/point.h"


int max_index(const std::vector<int> &vec);
float distance_from_plane(Point3D p, const std::vector<float> &coeffs, float sqrt);
Point3D cross_product(const Point3D &u, const Point3D &v);

std::vector<float> plane_coeffs(
    const Point3D &vec, 
    const Point3D &p1, 
    const Point3D &p2);

int get_close_points_indices(
    int index, 
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    std::vector<int> &close_points_indices);

int count_close_points(
    int index, 
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon);

int search_largest_plane(
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    int iter_num,
    float epsilon,
    std::vector<int> &plane_points_indices,
    int &num_of_close_points);

#endif