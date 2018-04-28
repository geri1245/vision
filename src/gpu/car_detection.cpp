#include "grid.h"
#include "car_detection.h"

//Temporary
#include "../util/input.h"
#include "../util/point.h"

Index get_index(const Point3D &coord, const Point3D &min, float cell_size)
{
    return { (int) ((coord.z - min.z) / cell_size),
             (int) ((coord.x - min.x) / cell_size) };
}

Point3D get_coords(const Index &ind, const Point3D &min, float cell_size)
{
    return 
    {
        min.x + ind.col * cell_size,
        -0.2,
        min.z + ind.row * cell_size
    };
}

bool check_and_set_neighbors(RolledGrid &grid, int thresh, Index &ind)
{
    int max = grid.at(ind.row, ind.col);
    if(max < thresh) //The current cell's value is smaller than the threshhold 
        return false;

    int good_neighbors = 0;
    int current;

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

std::vector<Point3D> detect_cars(const std::vector<Point3D> &points)
{/*
    constexpr int frame = 447;
    DirInputReader ir;
    ir.set_path("../../data1", "lidar1.xyz");
    
    for(int i = 0; i < frame; ++i)
        ir.step();

    std::vector<Point3D> points{ ir.next() };
*/
    constexpr Point3D min{-5, -0.55, -12};
    constexpr Point3D max{3, 0.4, 10};
    constexpr float cell_size = 1.0;
    constexpr int thresh = 150;

    std::cout << get_coords(get_index(min, min, cell_size), min, cell_size);
    std::cout << get_coords(get_index(max, min, cell_size), min, cell_size);

    constexpr int height = (max.z - min.z) / cell_size;
    constexpr int width  = (max.x - min.x) / cell_size;

    RolledGrid grid( height, width );
    /*grid.at(
        get_index( {-0.55, 0, -0.4}, min, cell_size)
    ) = 9999;
*/
    for(const auto &p : points)
    {
        if(    p > min && p < max &&
           !(( p.x < 0.3 && p.x > -0.9 ) &&
            (  p.z < 1.1 && p.z > -0.6 ) ) 
          )
            ++grid.at( get_index(p, min, cell_size) );
    }

    std::vector<Index> car_indices;
    car_indices.reserve(5);
    std::vector<Point3D> car_points;

    Index ind;
    
    for(int i = 1; i < height - 2; ++i)
    {
        for(int j = 1; j < width - 2; ++j)
        {
            ind = {i, j};
            if(check_and_set_neighbors(grid, thresh, ind))
            {
                car_indices.push_back(ind);
            }
        }
    }

    car_points.reserve(car_indices.size());

    for(const auto &ind : car_indices)
    {
        car_points.push_back( 
            { 
                get_coords(ind, min, cell_size)
            } 
        );
    }

    //grid.print();

    return std::move(car_points);
}

