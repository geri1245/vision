#include <random>
#include <iostream>
#include <algorithm>

#include "cpu_ransac_prep.h"


namespace{

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


//Returns the number of points that are within epsilon
//distance of the plane created by points[ randoms[index] ],
//points[ randoms[index + 1] ], and the direction up ({0, 1, 0}).
//Also populates close_points_indices vector, that contains
//the indices of the points close enough to the plane.
int get_close_points_indices(
    int index, 
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon,
    std::vector<int> &close_points_indices)
{
    const Point3D p1 = points[ randoms[2 * index] ];
    const Point3D p2 = points[ randoms[2 * index + 1] ];
    const Point3D up = {0, 1, 0};

    const std::vector<float> coeffs = plane_coeffs(up, p1, p2);
    float tmp_sqrt = sqrt(      //sqrt(A^2 + B^2 + C^2)
        coeffs[0] * coeffs[0] + 
        coeffs[1] * coeffs[1] + 
        coeffs[2] * coeffs[2]);
    
    close_points_indices.clear();
    close_points_indices.reserve(300);
    int num_of_close_points = 0;

    for(unsigned int i = 0; i < points.size(); ++i)
    {
        if(distance_from_plane(points[i], coeffs, tmp_sqrt) < epsilon)
        {
            ++num_of_close_points;
            close_points_indices.push_back(i);
        }
    }

    return num_of_close_points;
}

//Same as above, but doesn't store the point indices
int count_close_points(
    int index, 
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    float epsilon)
{
    const Point3D p1 = points[ randoms[2 * index] ];
    const Point3D p2 = points[ randoms[2 * index + 1] ];
    const Point3D up = {0, 1, 0};

    const std::vector<float> coeffs = plane_coeffs(up, p1, p2);
    float tmp_sqrt = sqrt(      //sqrt(A^2 + B^2 + C^2)
        coeffs[0] * coeffs[0] + 
        coeffs[1] * coeffs[1] + 
        coeffs[2] * coeffs[2]);
    
    int num_of_close_points = 0;

    for(size_t i = 0; i < points.size(); ++i)
    {
        if(distance_from_plane(points[i], coeffs, tmp_sqrt) < epsilon)
        {
            ++num_of_close_points;
        }
    }

    return num_of_close_points;
}

struct ComparePointByXAndZ
{
	bool operator()(const Point3D &lhs, const Point3D &rhs)
	{
		if( lhs.x < rhs.x )
		{
			return true;
		}
		else if ( lhs.x > rhs.x )
		{
			return false;
		}
		else
		{
			return lhs.z < rhs.z;
		}
	}
};


//Returns an index "a" for which 
//points[ random[a] ] and points[ random[a + 1] ] make 
//the largest plane from points 
//Also sets num_of_close_points to the number of points close enough
//And populates plane_points_indices with the indices of the points
int search_largest_plane(
    const std::vector<Point3D> &points,
    const std::vector<int> &randoms,
    int iter_num,
    float epsilon,
    std::vector<int> &plane_points_indices,
    int &num_of_close_points)
{
    int max_ind = 0, max_num = 0;
    int local_num_of_close_points;

    for(int i = 0; i < iter_num; i +=2)
    {
        local_num_of_close_points = count_close_points(
            i,
            points,
            randoms,
            epsilon
        );
        if(local_num_of_close_points > max_num)
        {
            max_num = local_num_of_close_points;
            max_ind = i;
        }
    }

    plane_points_indices.clear();
    plane_points_indices.reserve(max_num);

    num_of_close_points = get_close_points_indices(
        max_ind,
        points,
        randoms,
        epsilon,
        plane_points_indices
    );

    return max_ind;
}

}

std::vector<int> find_plane(
    const std::vector<Point3D> &points, 
    int iter_num, float epsilon)
{/*
    //Parameters:
    const int iter_num = 5000;
    std::vector<Point3D> points;
    const std::vector<int> gpu_sum_result;
    const float epsilon = 0.015;

    DirInputReader in;
    in.set_path("../../data1", "fusioned_no_color.xyz");
    std::vector<cv::Mat> a;
    points = in.next(a);
    const int num_of_points = points.size();
    */
    std::vector<int> randoms = get_random_numbers(2 * iter_num, 0, points.size() - 1);
    
    int max_ind;// = 70;//max_index(gpu_sum_result);
    int num_of_close_points;
    std::vector<int> close_points_indices;
    close_points_indices.reserve(700);

    max_ind = search_largest_plane(
        points,
        randoms,
        iter_num,
        epsilon,
        close_points_indices,
        num_of_close_points
    );

    /*
    std::cout << num_of_close_points << "\n" <<
        points[ randoms[2 * max_ind] ] <<
        points[ randoms[2 * max_ind + 1] ] << "\n";
    
    std::vector<Point3D> vec;
    vec.reserve(num_of_close_points);
    for(auto n : close_points_indices)
    {
        vec.push_back(points[n]);
    }
    
    std::sort(vec.begin(), vec.end(), ComparePointByXAndZ());

    for(const auto &p : vec)
    {
        //std::cout << p;
    }
    */
    return close_points_indices;
}