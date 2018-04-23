#include "cpu_math.h"

#include <algorithm>

int max_index(const std::vector<int> &vec)
{
    return std::distance(
        vec.begin(), 
        std::max_element(vec.begin(), vec.end())
    );
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
    close_points_indices.reserve(250);
    int num_of_close_points = 0;

    for(unsigned int i = 0; i < points.size(); ++i)
    {
        Point3D tmp = points[i];
        if( (tmp.x > 1.2 || tmp.x < -0.2) &&
            (tmp.z > 0.3 || tmp.z < -1.1) &&
            distance_from_plane(tmp, coeffs, tmp_sqrt) < epsilon)
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