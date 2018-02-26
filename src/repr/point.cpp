#include "point.h"

Point3D::Point3D(float x_, float y_, float z_) :
    x(x_), y(y_), z(z_) {}

Point3D::~Point3D() {}

Point3D::operator glm::vec3() const
{
    return glm::vec3{x, y, z};
}

Point3D operator+(const Point3D &lhs, const Point3D &rhs)
{
    return Point3D(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

Point3D operator-(const Point3D &lhs, const Point3D &rhs)
{
    return Point3D(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

bool operator==(const Point3D &lhs, const Point3D &rhs)
{
    return lhs.x == rhs.x &&
           lhs.y == rhs.y &&
           lhs.z == rhs.z;
}

Point3D operator*(const Point3D &lhs, float rhs)
{
    return Point3D(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

Point3D operator/(const Point3D &lhs, float rhs)
{
    return Point3D(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

Point3D operator+(const Point3D &lhs, float rhs)
{
    return Point3D(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}

Point3D operator-(const Point3D &lhs, float rhs)
{
    return Point3D(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}

std::ostream& operator<<(std::ostream &os, const Point3D &point)
{
    return os << "x = "   << point.x << 
                 ", y = " << point.y << 
                 ", z = " << point.z << "\n";
}

std::istream& operator>>(std::istream &in, Point3D &point)
{
    in >> point.x >> point.y >> point.z;
    return in;
}