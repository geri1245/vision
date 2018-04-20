#include "cam_calibration.h"

#include <string>
#include <fstream>

void read_mat3(std::istream &in, glm::mat3 &m)
{
    for(int i = 0; i < 9; ++i)
    {
        in >> m[i % 3][i / 3];
    }
}

void print_mat3(std::ostream &os, const glm::mat3 &m)
{
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            os << m[j][i] << " ";
        }
        os << "\n";
    }
}


MatVec::MatVec(const glm::mat3 &R_, const glm::vec3 &t_) :
    R(R_),
    t(t_),
    is_valid(true)
{}

MatVec::MatVec() :
    is_valid(false)
{}

glm::vec3 MatVec::mult_add(const glm::vec3 &v)
{
    return is_valid ? glm::vec3{R * v + t} : glm::vec3{};
}

ImageCalibration::ImageCalibration(int num, int size) : num { num }
{
    mat_vecs.resize(size);
}

glm::vec2 ImageCalibration::get_pixel_coords(const Point3D &p, int lidar_id)
{
    glm::vec3 tmp;
    double u, v;
    int id = lidar_id;

    //Find the correct lidar matrix and vector
    if( mat_vecs[lidar_id].is_valid )
    {
        id = lidar_id;
    }
    else
    {
        id = 1 - lidar_id;
    }

    tmp = mat_vecs[id].mult_add(to_vec3(p));
    tmp[0] /= tmp[2];
    tmp[1] /= tmp[2];

    float r_sq = tmp[0] * tmp[0] + tmp[1] * tmp[1];

    tmp[0] = tmp[0] * (1 + discoeff[0] * r_sq + discoeff[1] * r_sq * r_sq);
    tmp[1] = tmp[1] * (1 + discoeff[0] * r_sq + discoeff[1] * r_sq * r_sq);

    u = cam_mat[0][0] * tmp[0] + cam_mat[0][2];
    v = cam_mat[1][1] * tmp[1] + cam_mat[1][2];

    return glm::vec2{u, v};
}

CamCalibration::CamCalibration(int size)
{
    image_calibrations.reserve(size);
}

std::istream& operator>>(std::istream &in, ImageCalibration &im)
{
    std::string eq, name;
    while(in >> eq)
    {
        if(eq == "=")
        {
            if (name == "cammatrix")
            {
                read_mat3(in, im.cam_mat);
            }
            else if(name == "discoeff")
            {
                read_vec(in, im.discoeff, 5);
            }
        }
        else if(eq == "lidar1")
        {
            in >> im.mat_vecs[0];
        }
        else if(eq == "lidar2")
        {
            in >> im.mat_vecs[1];
        }
        else if(eq == "next")
        {
            break;
        }
        name = std::move(eq);
    }

    return in;
}

std::ostream &operator<<(std::ostream &os, const ImageCalibration &im)
{
    os << "camera matrix: ";
    print_mat3(os, im.cam_mat);
    os << "discoeff: ";
    print_vec(os, im.discoeff, 5);

    for(const auto &mv : im.mat_vecs)
    {
        os << mv << "\n";
    }

    return os;
}

std::istream& operator>>(std::istream &in, CamCalibration &cam)
{
    std::string name, eq;
    in >> cam.lidar1_to_lidar2;

    while(in >> eq)
    {
        if(eq == "Cam")
        {
            int num;
            in >> num;
            ImageCalibration im(num, 2);
            in >> im;
            cam.image_calibrations.push_back(im);
        }
    }

    return in;
}

std::ostream &operator<<(std::ostream &os, const CamCalibration &cam)
{
    os << "lidar1 to lidar 2: ";
    os << cam.lidar1_to_lidar2;
    for(unsigned int i = 0; i < cam.image_calibrations.size(); ++i)
    {
        os << "cam nr " << i << ":\n " << cam.image_calibrations[i]; 
    }

    return os;
}

std::ostream &operator<<(std::ostream &os, const MatVec &mv)
{
    if(mv.is_valid)
    {
        os << "R: ";
        print_mat3(os, mv.R);
        os << "t: ";
        print_vec(os, mv.t, 3);
    }

    return os;
}

std::istream &operator>>(std::istream &in, MatVec &mv)
{
    std::string eq, name;
    while(in >> eq)
    {
        if(eq == "=")
        {
            if(name == "R")
            {
                read_mat3(in, mv.R);
            }
            if(name == "t")
            {
                read_vec(in, mv.t, 3);
                break;
            }
        }

        name = std::move(eq);
    }

    mv.is_valid = true;

    return in;
}