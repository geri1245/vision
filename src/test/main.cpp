#define CATCH_CONFIG_MAIN

#include "../3rd/catch.hpp" 


#include "../repr/point.h"

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