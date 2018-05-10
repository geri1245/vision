#ifndef OBJECT_DETECTION_CAR_DETECTION_H
#define OBJECT_DETECTION_CAR_DETECTION_H

#include <vector>

#include "../input/point.h"

std::vector<Point3D> detect_cars(
    const std::vector<Point3D> &points,
    int lower_thresh = 50,
    int upper_thresh = 800);

#endif
