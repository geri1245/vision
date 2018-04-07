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

void print_mat3(const glm::mat3 &m)
{
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            std::cout << m[j][i] << " ";
        }
        std::cout << "\n";
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

CamCalibration::CamCalibration(int size)
{
    image_calibrations.reserve(size);
}

void CamCalibration::add_image_calibration(const ImageCalibration &image_cal)
{
    image_calibrations.push_back(image_cal);
}

void CamCalibration::set_Rt(glm::mat3 R_, glm::vec3 t_)
{
    lidar1_to_lidar2 = MatVec(R_, t_);
}

/*
int main()
{
    std::ifstream in{"../../data1/calibration.txt"};
    std::string lidar, name, eq;
    int i = 0;

    glm::mat3 R;
    glm::vec3 t;

    in >> eq;
    while(eq != "Cam")
    {
        if(eq == "=")
        {
            if(name == "R")
            {
                read_mat3(in, R);
            }
            if(name == "t")
            {
                read_vec(in, t, 3);
                break;
            }
        }

        name = std::move(eq);
        in >> eq;
    }

    print_mat3(R);
    print_vec(t, 3);

    in >> i;

    name = std::move(eq);

    while(in >> eq)
    {
        if(eq == "=")
        {
            if (name == "cammatrix")
            {}
            else if(name == "discoeff")
            {}
            else if(name == "R")
            {}
            else if(name == "t")
            {}
            std::cout << lidar << " " << name << eq << "\n";
        }
        else if(eq == "Cam")
        {
            
        }
        

        //std::cout << i++ << ". : " << eq << "\n";

        lidar = std::move(name);
        name = std::move(eq);
    }

    std::cout << eq;

    return 0;
}*/