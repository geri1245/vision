#ifndef UTIL_CAM_CALIB_H 
#define UTIL_CAM_CALIB_H 

#include <iostream>
#include <optional>
#include <array>
#include <vector>

#include <glm/glm.hpp>

struct ImageCalibration
{
    int num;
    glm::mat3 cam_mat;
    std::array<float, 5> discoeff;
    std::optional<glm::mat3> R1;
    std::optional<glm::vec3> t1;
    std::optional<glm::mat3> R2;
    std::optional<glm::vec3> t2;
};

class CamCalibration
{
    CamCalibration(int size);
    void add_image_calibration(const ImageCalibration &image_cal);
    void set_Rt(glm::mat3 R_, glm::vec3 t_);

private:
    std::vector<ImageCalibration> image_calibrations;
    glm::mat3 R;
    glm::vec3 t;
};

std::istream& operator>>(std::istream &in, ImageCalibration &im);

#endif