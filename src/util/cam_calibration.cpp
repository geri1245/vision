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

template <typename T>
void read_vec(std::istream &in, T &vec, int size)
{
    for(int i = 0; i < size; ++i)
    {
        in >> vec[i];
    }
}

void print(const glm::mat3 &m)
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

template <typename T>
void print(const T &vec, int size)
{
    for(int i = 0; i < size; ++i)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
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
    R = R_;
    t = t_;
}

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

    print(R);
    print(t, 3);

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
}