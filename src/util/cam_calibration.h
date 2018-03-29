#ifndef UTIL_CAM_CALIB_H 
#define UTIL_CAM_CALIB_H 

#include <iostream>

#include <glm/glm.hpp>

struct CameraCalibration
{
    int num;
    glm::mat3 cam_mat;
    glm::mat3 R;
    glm::vec3 t;

};

std::istream& operator>>(std::istream &in, CameraCalibration &cam);

#endif