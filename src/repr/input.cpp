#include "input.h"

InputReader::InputReader(const std::string &filename_, int size_) : 
    size(size_),
    filename(filename_), 
    in(filename.c_str())
    {}

InputReader::~InputReader() {}

std::vector<Point3D> InputReader::get_points()
{
    std::vector<Point3D> ret;
    ret.reserve(size);
    Point3D p;

    while(in >> p)
    {
        ret.push_back(p);
    }

    return ret;
}