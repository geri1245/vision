#Test makefile

cc = g++
libs = -lSDL2 -lSDL2main -lGL -lGLU -lglut -lGLEW -lstdc++fs
cflags = -Wall -Wextra -std=c++17

deps = ../util/*.cpp
cpps := $(shell find -name '*.cpp')
objects := $(cpps:%.cpp=%.o)

.PHONY : all
test : $(objects)
	$(cc) $(objects) $(deps) -o $@ $(libs) $(cflags)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(cflags)
