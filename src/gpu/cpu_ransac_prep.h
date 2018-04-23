#ifndef GPU_CPU_RANSAC_PREP_H
#define GPU_CPU_RANSAC_PREP_H

#include <vector>

#include "../util/point.h" 

std::vector < std::vector<int> > find_plane(
    const std::vector<Point3D> &points, 
    int iter_num = 5000, float epsilon = 0.015);

#endif