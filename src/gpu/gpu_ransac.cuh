#ifndef GPU_GPU_RANSAC_CUH
#define GPU_GPU_RANSAC_CUH

#include <vector>
#include "../util/point_raw.h"

int get_points_close_to_plane(
    int index, 
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    std::vector<int> &close_points_indices);

#endif