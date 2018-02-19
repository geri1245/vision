#ifndef REPR_POINT_H
#define REPR_POINT_H

#include <iostream>
#include <fstream>

struct Point3D
{
    Point3D(float x_, float y_, float z_);
    ~Point3D();

    float x, y, z;
};

Point3D operator+(const Point3D &lhs, const Point3D &rhs);
Point3D operator-(const Point3D &lhs, const Point3D &rhs);
Point3D operator*(const Point3D &lhs, float rhs);
Point3D operator/(const Point3D &lhs, float rhs);
Point3D operator+(const Point3D &lhs, float rhs);
Point3D operator-(const Point3D &lhs, float rhs);

std::ostream& operator<<(std::ostream &os, const Point3D &point);



#endif