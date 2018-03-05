#define CATCH_CONFIG_MAIN

#include "../3rd/catch.hpp" 


#include "../util/point.h"
#include "../util/input.h"

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
    ir.set_filename( "test/testdata/1/a.test" );

    std::vector<Point3D> points{ ir.get_points() };

    Point3D a{ 5, 7, 4 };
    Point3D b{ 6, -9, 4 };
    Point3D c{ 12, 784, 54.1 };

    REQUIRE( points[0] == a );
    REQUIRE( points[1] == b );
    REQUIRE( points[2] == c );
}

TEST_CASE( "Testing DirInputReader class." "[DirInputReader]" )
{
    DirInputReader dir{ "test/testdata", "a.test" };
    std::vector<Point3D> p1{ dir.next() };

    dir.step();

    std::vector<Point3D> p2{ dir.next() };

    bool should_be_false = dir.step();

    Point3D a{ 5, 7, 4 };
    Point3D b{ 6, -9, 4 };
    Point3D c{ 12, 784, 54.1 };

    Point3D d{ 78, 894, -97.4 };
    Point3D e{ 4, 7, 199779 };

    REQUIRE( p1[0] == a );
    REQUIRE( p1[1] == b );
    REQUIRE( p1[2] == c );

    REQUIRE( p2[0] == d );
    REQUIRE( p2[1] == e );
    REQUIRE( !should_be_false );
}