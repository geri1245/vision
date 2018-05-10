#ifndef OBJECT_DETECTION_GRID_H
#define OBJECT_DETECTION_GRID_H

#include <vector>
#include <iostream>

struct Index
{
    int row, col;
};

std::ostream& operator<< (std::ostream &out, const Index &ind);

struct RolledGrid
{
public:

    RolledGrid(int height, int width);
    int& at(int row, int col);
    int& at(const Index &ind);
    void print(std::ostream &out = std::cout);

    int height, width;
    int not_valid_accesses;
    std::vector<int> grid;
};

#endif