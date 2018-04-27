#include <random>
#include <iostream>

#include "cpu_ransac_prep.h"
#include "../util/input.h"
#include "gpu_ransac.cuh"
#include "cpu_math.h"

namespace
{
    std::vector<int> get_random_numbers(int num, int min, int max)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(min, max);

    std::vector<int> randoms;
    randoms.reserve(num);

    for (int i = 0; i < num; ++i)
        randoms.push_back( dist(mt) );

    return randoms;
}
}


std::vector< std::vector<Point3D> > find_plane(
    const std::vector<Point3D> &points, 
    int iter_num, float epsilon, int threshhold)
{
    std::vector<int> randoms = get_random_numbers(2 * iter_num, 0, points.size() - 1);
    
    std::vector<std::vector<int> > plane_points_indices;
    plane_points_indices.reserve(10);

    get_points_close_to_plane(
            iter_num,
            points,
            randoms,
            epsilon,
            threshhold,
            plane_points_indices
        );

    std::vector< std::vector<Point3D> > plane_points;
    plane_points.resize(plane_points_indices.size());
    for(const auto &v : plane_points_indices)
    {
        std::vector<Point3D> tmp_points;
        tmp_points.clear();
        tmp_points.reserve(v.size());
        for(int ind : v)
        {
            tmp_points.push_back(points[ind]);
        }
        plane_points.push_back(std::move(tmp_points));
    }

    return std::move(plane_points);
}