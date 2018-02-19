#include "repr/point.h"
#include "input/input.h"

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    Point3D a(4, 5, 7);
    Point3D b(1, 2, 11);

    std::vector<int> v;

    std::cout << a + b;

    return 0;
}