#ifndef GPU_CPU_RANSAC_PREP_H
#define GPU_CPU_RANSAC_PREP_H

#include <vector>

#include "../input/point.h" 

std::vector < std::vector<Point3D> > find_plane(
    const std::vector<Point3D> &points, 
    int iter_num = 5000, float epsilon = 0.015,
    int threshhold = 190);

#endif