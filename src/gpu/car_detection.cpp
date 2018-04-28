#include "grid.h"
#include "car_detection.h"

//Temporary
#include "../util/input.h"
#include "../util/point.h"

float get_index(float coord, float min, float cell_size)
{
    return (coord - min) / cell_size;
}

bool check_and_set_neighbors(RolledGrid &grid, int thresh, Index &ind)
{
    int max = grid.at(ind.row, ind.col);
    if(max < thresh) //The current cell's value is smaller than the threshhold 
        return false;

    int good_neighbors = 0;
    int current, max_ind;

    std::vector<Index> neighbors;
    neighbors.reserve(8);
    
    neighbors.push_back( {ind.row - 1, ind.col - 1} );
    neighbors.push_back( {ind.row    , ind.col - 1} );
    neighbors.push_back( {ind.row + 1, ind.col - 1} );
    neighbors.push_back( {ind.row - 1, ind.col    } );
    neighbors.push_back( {ind.row + 1, ind.col    } );
    neighbors.push_back( {ind.row - 1, ind.col + 1} );
    neighbors.push_back( {ind.row    , ind.col + 1} );
    neighbors.push_back( {ind.row + 1, ind.col + 1} );

    for(int i = 0; i < 8; ++i)
    {
        current = grid.at(neighbors[i]);
        if(current > thresh) //Same col
        {
            ++good_neighbors;
            if(current > max)
            {
                ind = neighbors[i];
                max = current;
            }
            grid.at(neighbors[i]) = 0; //We don't want to count it again
        }
    }

    if(good_neighbors > 0)
    {
        return true;
    }
    return false;
}

int main()
{
    constexpr int frame = 261;
    DirInputReader ir;
    
    for(int i = 0; i < frame; ++i)
        ir.step();

    ir.set_path("../../data1", "lidar1.xyz");
    std::vector<Point3D> points{ ir.next() };

    constexpr Point3D min{-5, -0.55, -12};
    constexpr Point3D max{2, 0.4, 10};
    constexpr float cell_size = 0.5;

    constexpr int height = (max.z - min.z) / cell_size;
    constexpr int width  = (max.x - min.x) / cell_size;

    RolledGrid grid( height, width );
    grid.at(
        get_index(-0.55, min.z, cell_size),
        get_index(-0.4, min.x, cell_size)
    ) = 9999;

    for(const auto &p : points)
    {
        if(    p > min && p < max &&
           !(( p.x < 0.2 && p.x > -0.9 ) &&
            (  p.z < 1.1 && p.z > -0.3 ) ) 
          )
            ++grid.at(
                get_index(p.z, min.z, cell_size),
                get_index(p.x, min.x, cell_size)
            );
    }

    std::vector<Index> car_indices;
    car_indices.reserve(5);
    Index ind;
    for(int i = 1; i < height - 2; ++i)
    {
        for(int j = 1; j < width - 2; ++j)
        {
            ind = {i, j};
            if(check_and_set_neighbors(grid, 200, ind))
            {
                car_indices.push_back(ind);
            }
        }
    }

    for(const auto &ind : car_indices)
    {
        grid.at(ind) = 777;
        std::cout << ind;
    }

    grid.print();

    return 0;
}

