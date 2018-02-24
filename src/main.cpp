#include "init.h"
#include "repr/debug.hpp"
#include "repr/input.h"

int main()
{
    //initialize();

    InputReader ir("../data/test1.asd");

    std::vector<Point3D> points{ir.get_points()};

    print(points);

    return 0;
}