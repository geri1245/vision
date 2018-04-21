#include <random>
#include <iostream>
#include <algorithm>

#include "../util/point.h"
#include "../util/input.h"

int max_index(const std::vector<int> &vec)
{
    return std::distance(
        vec.begin(), 
        std::max_element(vec.begin(), vec.end())
    );
}

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

float distance_from_plane(Point3D p, const std::vector<float> &coeffs, float sqrt)
{
    //d = abs( A*x1 + B*y1 + C*z1 ) / sqrt(A^2 + B^2 + C^2) 
    return 
        abs(coeffs[0] * p.x + coeffs[1] * p.y + coeffs[2] * p.z + coeffs[3]) /
        sqrt;    
}

Point3D cross_product(const Point3D &u, const Point3D &v)
{
    return {
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    };
}

std::vector<float> plane_coeffs(
    const Point3D &vec, 
    const Point3D &p1, 
    const Point3D &p2)
{
    std::vector<float> coeffs;
    coeffs.reserve(4);
    Point3D vec2 = p1 - p2; //Calculate another direction vector
    
    //Calculating the normal vector of the plane
    Point3D normal = cross_product(vec, vec2);

    coeffs.push_back(normal.x);
    coeffs.push_back(normal.y);
    coeffs.push_back(normal.z);
    coeffs.push_back(-normal.x * p1.x - normal.y * p1.y - normal.z * p1.z);

    return coeffs;
}

int get_close_points_indices(
    int index_of_max_elem, 
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    std::vector<int> &close_points_indices)
{
    const Point3D p1 = points[ randoms[2 * index_of_max_elem] ];
    const Point3D p2 = points[ randoms[2 * index_of_max_elem + 1] ];
    const Point3D up = {0, 1, 0};

    const std::vector<float> coeffs = plane_coeffs(up, p1, p2);
    float tmp_sqrt = sqrt(      //sqrt(A^2 + B^2 + C^2)
        coeffs[0] * coeffs[0] + 
        coeffs[1] * coeffs[1] + 
        coeffs[2] * coeffs[2]);
    
    close_points_indices.reserve(300);
    int num_of_close_points = 0;

    for(size_t i = 0; i < points.size(); ++i)
    {
        if(distance_from_plane(points[i], coeffs, tmp_sqrt) < epsilon)
        {
            ++num_of_close_points;
            close_points_indices.push_back(i);
        }
    }

    return num_of_close_points;
}

int main()
{
    //Parameters:
    const int num_of_points = 28000;
    const int iter_num = 10000;
    std::vector<Point3D> points;
    const std::vector<int> gpu_sum_result;
    const float epsilon = 0.01;

    DirInputReader in;
    in.set_path("../../data1", "fusioned_no_color.xyz");
    std::vector<cv::Mat> a;
    points = in.next(a);

    std::vector<int> randoms = get_random_numbers(2 * iter_num, 0, num_of_points - 1);
    
    int max_ind = 70;//max_index(gpu_sum_result);

    std::vector<int> close_points_indices;
    int num_of_close_points = get_close_points_indices(
        max_ind,
        points,
        randoms,
        epsilon,
        close_points_indices
    );
    
    std::cout << num_of_close_points << "\n" <<
        points[ randoms[2 * max_ind] ] <<
        points[ randoms[2 * max_ind +1] ] << "\n";
    
    for(auto n : close_points_indices)
    {
        std::cout << points[n];
    }
}