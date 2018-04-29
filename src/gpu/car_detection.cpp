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

bool check_and_set_neighbors(RolledGrid &grid, int lower_thresh, int upper_thresh, Index &ind)
{
    int max = grid.at(ind.row, ind.col);
    if(max < lower_thresh || max > upper_thresh) //The current cell's value is smaller than the threshhold 
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
        if(current > lower_thresh && current < upper_thresh) //Same col
        {
            ++good_neighbors;
            if(current > max)
            {
                ind = neighbors[i];
                max = current;
            }
            grid.at(neighbors[i]) = 70; //We don't want to count it again
        }
    }

    if(good_neighbors > 1)
    {
        return true;
    }
    return false;
}

bool check_and_set_neighbors2(RolledGrid &grid, int lower_thresh, int upper_thresh, Index &ind)
{
    int max = grid.at(ind.row, ind.col);
    if(max < lower_thresh) //The current cell's value is smaller than the threshhold 
        return false;

    int sum = 0;
    int current;

    for(int i = ind.row - 2; i < ind.row + 3; ++i)
    {
        for(int j = ind.col - 2; j < ind.col + 3; ++j)
        {
            current = grid.at(i, j);
            sum += current;
            
            if(current > max)
            {
                max = current;
                ind = {i, j};
            }
        }
    }
    
    if(sum > lower_thresh && sum < upper_thresh)
    {
        for(int i = ind.row - 2; i < ind.row + 3; ++i)
        {
            for(int j = ind.col - 2; j < ind.col + 3; ++j)
            {
                grid.at(i, j) = 0;
            }
        }
        return true;
    }
    return false;
}

std::vector<Point3D> detect_cars(const std::vector<Point3D> &points)
{
    //Parameters for car detection
    //Point3Ds specify the range of search
    constexpr Point3D min{-10, -0.55, -20};
    constexpr Point3D max{4, 0.4, 20};
    
    constexpr int lower_thresh = 50;
    constexpr int upper_thresh = 800;

    constexpr float cell_size = 0.5;
    constexpr int height = (max.z - min.z) / cell_size;
    constexpr int width  = (max.x - min.x) / cell_size;

    RolledGrid grid( height, width );

    for(const auto &p : points)
    {
        if(    p > min && p < max
           
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
            if(check_and_set_neighbors2(grid, lower_thresh, upper_thresh, ind))
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

    return std::move(car_points);
}

