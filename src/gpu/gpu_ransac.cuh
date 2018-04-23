#ifndef GPU_GPU_RANSAC_CUH
#define GPU_GPU_RANSAC_CUH

#include <vector>
#include "../util/point_raw.h"

int get_points_close_to_plane(
    int iter_num,
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    int threshhold,
    std::vector< std::vector<int> > &plane_points);

#endif