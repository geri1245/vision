#ifndef REPR_POINT_H
#define REPR_POINT_H

#include <iostream>
#include <fstream>

#include <glm/glm.hpp>

#include "point_raw.h"

struct ComparePointByXAndZ
{
	bool operator()(const Point3D &lhs, const Point3D &rhs)
	{
		if( lhs.x < rhs.x )
		{
			return true;
		}
		else if ( lhs.x > rhs.x )
		{
			return false;
		}
		else
		{
			return lhs.z < rhs.z;
		}
	}
};

struct ComparePointByZAndX
{
	bool operator()(const Point3D &lhs, const Point3D &rhs)
	{
		if( lhs.z < rhs.z )
		{
			return true;
		}
		else if ( lhs.z > rhs.z )
		{
			return false;
		}
		else
		{
			return lhs.x < rhs.x;
		}
	}
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 col;
};

Point3D operator+ (const Point3D &lhs, const Point3D &rhs);
Point3D operator- (const Point3D &lhs, const Point3D &rhs);
bool    operator==(const Point3D &lhs, const Point3D &rhs);

Point3D operator* (const Point3D &lhs, float rhs);
Point3D operator/ (const Point3D &lhs, float rhs);
Point3D operator+ (const Point3D &lhs, float rhs);
Point3D operator- (const Point3D &lhs, float rhs);

glm::vec3 to_vec3(const Point3D &p);

std::ostream& operator<<(std::ostream &os, const Point3D &point);
std::istream& operator>>(std::istream &in, Point3D &point);

#endif