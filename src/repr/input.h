#ifndef INPUT_INPUT_H
#define INPUT_INPUT_H

#include <fstream>
#include <string>
#include <vector>

#include "point.h"

class InputReader
{
public:

    InputReader(const std::string &filename_, int size = 10000);
    ~InputReader();
    std::vector<Point3D> get_points();

private:

    void read_data();

    int size;
    const std::string filename;
    std::ifstream in;
};

#endif