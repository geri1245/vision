#include "grid.h"

std::ostream& operator<< (std::ostream &out, const Index &ind)
{
    return out << "row: " << ind.row << ", col: " << ind.col << "\n";
}


RolledGrid::RolledGrid(int height, int width) : 
    height(height), width(width), not_valid_accesses{0}, grid(width * height, 0)
{
}

int& RolledGrid::at(int row, int col)
{
    return grid[row * width + col];
}

int& RolledGrid::at(const Index &ind)
{
    return grid[ind.row * width + ind.col];
}

void RolledGrid::print(std::ostream &out)
{
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            out << grid[i * width + j] << " ";
        }
        out << "\n";
    }
}