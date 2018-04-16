#ifndef REPR_POINT_H
#define REPR_POINT_H

#include <iostream>
#include <fstream>

#include <glm/glm.hpp>

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 col;
    glm::vec2 tex;
};

struct Point3D
{
    Point3D(float x_ = 0.f, float y_ = 0.f, float z_ = 0.f);
    ~Point3D();

    operator glm::vec3() const;

    float x, y, z;
};

Point3D operator+ (const Point3D &lhs, const Point3D &rhs);
Point3D operator- (const Point3D &lhs, const Point3D &rhs);
bool    operator==(const Point3D &lhs, const Point3D &rhs);

Point3D operator* (const Point3D &lhs, float rhs);
Point3D operator/ (const Point3D &lhs, float rhs);
Point3D operator+ (const Point3D &lhs, float rhs);
Point3D operator- (const Point3D &lhs, float rhs);

std::ostream& operator<<(std::ostream &os, const Point3D &point);
std::istream& operator>>(std::istream &in, Point3D &point);

#endif