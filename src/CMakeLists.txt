cmake_minimum_required(VERSION 3.0)
project( main )

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
                          -std=c++17 \
                          -Wall \
                          -Wextra")

set(sources 
    main.cpp repr/point.cpp input/input.cpp
)

add_executable( main ${sources} )

