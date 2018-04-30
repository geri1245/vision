#ifndef GPU_CAR_DETECTION_H
#define GPU_CAR_DETECTION_H

#include <vector>

#include "../input/point.h"

std::vector<Point3D> detect_cars(const std::vector<Point3D> &points);

#endif