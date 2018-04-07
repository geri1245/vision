#ifndef UTIL_CAM_CALIB_H 
#define UTIL_CAM_CALIB_H 

#include <iostream>
#include <optional>
#include <array>
#include <vector>

#include <glm/glm.hpp>

class MatVec
{
public:
    MatVec();
    MatVec(const glm::mat3 &R_, const glm::vec3 &t_);
    glm::vec3 mult_add(const glm::vec3 &v);

//private:
    glm::mat3 R;
    glm::vec3 t;
    bool is_valid;
};

class ImageCalibration
{
public:
    ImageCalibration(int num);

private:
    int num;
    glm::mat3 cam_mat;
    std::array<float, 5> discoeff;
    std::vector<MatVec> mat_vecs;
};

class CamCalibration
{
    CamCalibration(int size);
    void add_image_calibration(const ImageCalibration &image_cal);
    void set_Rt(glm::mat3 R_, glm::vec3 t_);

private:
    std::vector<ImageCalibration> image_calibrations;
    MatVec lidar1_to_lidar2;
};

void read_mat3(std::istream &in, glm::mat3 &m);
void print_mat3(const glm::mat3 &m);

template <typename T>
void read_vec(std::istream &in, T &vec, int size)
{
    for(int i = 0; i < size; ++i)
    {
        in >> vec[i];
    }
}

template <typename T>
void print_vec(const T &vec, int size)
{
    for(int i = 0; i < size; ++i)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
}


std::istream& operator>>(std::istream &in, ImageCalibration &im);

#endif