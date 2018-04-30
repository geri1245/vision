#ifndef GPU_CPU_RANSAC_PREP_H
#define GPU_CPU_RANSAC_PREP_H

#include <vector>

#include "../input/point.h" 

std::vector < std::vector<Point3D> > find_plane(
    const std::vector<Point3D> &points, 
    int iter_num = 5120, float epsilon = 0.002,
    int threshhold = 60);

#endif