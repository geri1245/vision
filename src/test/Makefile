#Test makefile

cc = g++
cflags = -Wall -Wextra -std=c++17

deps = ../repr/point.cpp
cpps := $(shell find -name '*.cpp')
objects := $(cpps:%.cpp=%.o)

.PHONY : all
test : $(objects)
	$(cc) $(objects) $(deps) -o $@ $(cflags)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(cflags)
