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


std::vector< std::vector<int> > find_plane(
//int main(
    const std::vector<Point3D> &points, 
    int iter_num, float epsilon, int threshhold)
{
/*int main()
{
    //Parameters:
    const int iter_num = 2048 * 10;
    std::vector<Point3D> points;
    const std::vector<int> gpu_sum_result;
    const float epsilon = 0.015;
    const int threshhold = 190;

    DirInputReader in;
    in.set_path("../../data1", "fusioned_no_color.xyz");
    std::vector<cv::Mat> a;
    points = in.next(a);*/
    
    std::vector<int> randoms = get_random_numbers(2 * iter_num, 0, points.size() - 1);
    
/*
    max_ind = search_largest_plane(
        points,
        randoms,
        iter_num,
        epsilon,
        close_points_indices,
        num_of_close_points
    );
*/
    std::vector<std::vector<int> > plane_points;
    plane_points.reserve(10);


    /*for(int i = 0; i < iter_num; ++i)
    {
        tmp.push_back(
            get_close_points_indices(
                i,
                points,
                randoms,
                epsilon,
                close_points_indices));
    }

    for(auto n : tmp)
    {
        std::cout << n << " ";
    }*/

    


    //GPU result:
    //std::cout << "GPU result:\n";
    get_points_close_to_plane(
            iter_num,
            points,
            randoms,
            epsilon,
            threshhold,
            plane_points
        );

    /*for(const auto &v : plane_points)
    {
        std::cout << "\nsize: " << v.size() << "\n";
        for(int i : v)
        {
            std::cout << i << " ";
        }
    }*/
    
    /*
    for(int i = 0; i < 2; ++i)
    {
        get_points_close_to_plane(
            iter_num,
            points,
            randoms,
            epsilon,
            threshhold,
            gpu_results
        );

        auto it = std::max_element(gpu_results.begin(), gpu_results.end());
        std::cout << "max num: " << *it << "\n";
        if(*it < threshhold)
            break; //Not a big enough plane, we stop

        std::vector<int> tmp;
        tmp.reserve(200);
        get_close_points_indices(
            std::distance(gpu_results.begin(), it), points,
            randoms, epsilon,
            tmp);
        plane_points.push_back(std::move(tmp));
    }

    for(const auto &v : plane_points)
    {
        std::cout << "\nsize:" << v.size() << "\n"; 
        std::cout << "indices: \n";
        for(int i : v)
        {
            std::cout << i << " ";
        }
    }*/

/*
    std::sort(gpu_results.begin(), gpu_results.end(), std::greater<int>());
    for(auto n : gpu_results)
    {
        std::cout << n << " ";
    }*/
    /*
    std::cout << num_of_close_points << "\n" <<
        points[ randoms[2 * max_ind] ] <<
        points[ randoms[2 * max_ind + 1] ] << "\n";
    
    vec.clear();
    vec.reserve(num_of_close_points);
    for(auto n : close_points_indices)
    {
        vec.push_back(points[n]);
    }
    
    std::sort(vec.begin(), vec.end(), ComparePointByXAndZ());

    for(const auto &p : vec)
    {
        std::cout << p;
    }
    */
    //return 0;
    return std::move(plane_points);//close_points_indices;
}