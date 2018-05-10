#define CATCH_CONFIG_MAIN

#include <fstream>

#include <glm/glm.hpp>

#include "../3rd/catch.hpp" 
#include "../input/point.h"
#include "../input/input.h"
#include "../colorer/cam_calibration.h"

TEST_CASE( "Testing Point3D class.", "[Point3D]" )
{
    Point3D p1{1, 2, 7};
    Point3D p2{8, 8, 10};

    REQUIRE( p1 + p2  == Point3D{9.0, 10.0, 17.0} );
    REQUIRE( p2 - p1  == Point3D{7.0, 6.0, 3.0} );
    REQUIRE( p1 + 3   == Point3D{4.0, 5.0, 10.0} );
    REQUIRE( p1 - 12  == Point3D{-11.0, -10.0, -5.0} );
    REQUIRE( p1 * 3   == Point3D{3, 6, 21} );
}

TEST_CASE( "Testing InputReader class." "[InputReader]" )
{
    InputReader ir{};
    ir.set_filename( "testdata/1/a.test" );

    std::vector<Point3D> points{ ir.get_points() };

    Point3D a{ 5, 4, 7 };
    Point3D b{ 6, 4, -9 };
    Point3D c{ 12, 54.1, 784 };

    REQUIRE( points[0] == a );
    REQUIRE( points[1] == b );
    REQUIRE( points[2] == c );
}

TEST_CASE( "Testing DirInputReader class." "[DirInputReader]" )
{
    DirInputReader dir{ "testdata", "a.test" };
    std::vector<Point3D> p1{ dir.get_data() };

    dir.step();

    std::vector<Point3D> p2{ dir.get_data() };

    bool should_be_false = dir.step();

    Point3D a{ 5, 4, 7 };
    Point3D b{ 6, 4, -9 };
    Point3D c{ 12, 54.1, 784 };

    Point3D d{ 78, -97.4, 894 };
    Point3D e{ 4, 199779, 7 };

    REQUIRE( p1[0] == a );
    REQUIRE( p1[1] == b );
    REQUIRE( p1[2] == c );

    REQUIRE( p2[0] == d );
    REQUIRE( p2[1] == e );
    REQUIRE( !should_be_false );
}

TEST_CASE( "Testing MatVec class." "[MatVec]" )
{
    //Test 1
    std::ifstream in{ "test/mat_vec.test" };
    glm::mat3 mat;
    glm::vec3 vec1, vec2, result, expected_result;

    read_mat3(in, mat);
    read_vec(in, vec1, 3);
    read_vec(in, vec2, 3);
    read_vec(in, expected_result, 3);

    MatVec mv(mat, vec2);
    result = mv.mult_add(vec1);

    REQUIRE( expected_result.x == result.x );
    REQUIRE( expected_result.y == result.y );
    REQUIRE( expected_result.z == result.z );


    //Test2
    in.open( "test/mat_vec2.test" );

    read_mat3(in, mat);
    read_vec(in, vec1, 3);
    read_vec(in, vec2, 3);
    read_vec(in, expected_result, 3);

    mv = MatVec(mat, vec2);
    result = mv.mult_add(vec1);

    REQUIRE( expected_result.x == result.x );
    REQUIRE( expected_result.y == result.y );
    REQUIRE( expected_result.z == result.z );


    //Test3
    in.open( "test/mat_vec3.test" );

    read_mat3(in, mat);
    read_vec(in, vec1, 3);
    read_vec(in, vec2, 3);
    read_vec(in, expected_result, 3);

    mv = MatVec(mat, vec2);
    result = mv.mult_add(vec1);

    REQUIRE( expected_result.x == result.x );
    REQUIRE( expected_result.y == result.y );
    REQUIRE( expected_result.z == result.z );
}