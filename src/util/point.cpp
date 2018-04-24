#include "point.h"

glm::vec3 to_vec3(const Point3D &p)
{
    return glm::vec3{p.x, p.y, p.z};
}

Point3D operator+(const Point3D &lhs, const Point3D &rhs)
{
    return Point3D{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Point3D operator-(const Point3D &lhs, const Point3D &rhs)
{
    return Point3D{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

bool operator==(const Point3D &lhs, const Point3D &rhs)
{
    return lhs.x == rhs.x &&
           lhs.y == rhs.y &&
           lhs.z == rhs.z;
}

Point3D operator*(const Point3D &lhs, float rhs)
{
    return Point3D{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

Point3D operator/(const Point3D &lhs, float rhs)
{
    return Point3D{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

Point3D operator+(const Point3D &lhs, float rhs)
{
    return Point3D{lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
}

Point3D operator-(const Point3D &lhs, float rhs)
{
    return Point3D{lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
}

std::ostream& operator<<(std::ostream &os, const Point3D &point)
{
    return os << "x = "   << point.x << 
                 ", y = " << point.y << 
                 ", z = " << point.z << "\n";
}

std::istream& operator>>(std::istream &in, Point3D &point)
{
    in >> point.x >> point.z >> point.y;
    while(point.x == 0 && point.y == 0 && point.z == 0)
        in >> point.x >> point.z >> point.y;

    return in;
}