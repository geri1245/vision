#ifndef COLORER_CAM_CALIB_H 
#define COLORER_CAM_CALIB_H 

#include <iostream>
#include <optional>
#include <array>
#include <vector>

#include <glm/glm.hpp>

#include "../input/point.h"

struct CameraSelector
{
    int operator()(const Point3D &p)
    {
        Point3D norm_p = normalize(p);
        double x = norm_p.x;
        double z = norm_p.z;
        int cam = 0;
        
        
        if(x > 0)
        {
            if(x > cosa)
                cam = 0;
            else if(z > 0)
                cam = 5;
            else
                cam = 3;
        }

        else if(x <= 0)
        {
            if(x < -cosa)
                cam = 2;
            else if(z > 0)
                cam = 4;
            else 
                cam = 1;
        }

        return cam;
    }

private:

    Point3D normalize(const Point3D &p);

    float cosa = cos(M_PI * 4.5 / 18.0f); //cos(60)
};

struct MatVec
{
    MatVec();
    MatVec(const glm::mat3 &R_, const glm::vec3 &t_);
    glm::vec3 mult_add(const glm::vec3 &v);

    glm::mat3 R;
    glm::vec3 t;
    bool is_valid;
};

struct ImageCalibration
{
    ImageCalibration(int num, int size);
    glm::vec2 get_pixel_coords(const Point3D &p, int lidar_id);

    int num;
    glm::mat3 cam_mat;
    std::array<float, 5> discoeff;
    std::vector<MatVec> mat_vecs;
};

struct CamCalibration
{
    explicit CamCalibration(int size);    

    std::vector<ImageCalibration> image_calibrations;
    MatVec lidar1_to_lidar2;
};

void read_mat3(std::istream &in, glm::mat3 &m);
void print_mat3(std::ostream &os, const glm::mat3 &m);

template <typename T>
inline std::istream& read_vec(std::istream &in, T &vec, int size)
{
    for(int i = 0; i < size; ++i)
    {
        in >> vec[i];
    }

    return in;
}

template <typename T>
inline std::ostream& print_vec(std::ostream &os, const T &vec, int size)
{
    for(int i = 0; i < size; ++i)
    {
        os << vec[i] << " ";
    }
    os << "\n";

    return os;
}


std::istream& operator>>(std::istream &in, MatVec &mv);
std::istream& operator>>(std::istream &in, ImageCalibration &im);
std::istream& operator>>(std::istream &in, CamCalibration &cam);
std::ostream &operator<<(std::ostream &os, const MatVec &mv);
std::ostream &operator<<(std::ostream &os, const ImageCalibration &im);
std::ostream &operator<<(std::ostream &os, const CamCalibration &cam);

#endif