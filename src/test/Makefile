#Test makefile

cc = g++
libs = -lSDL2 -lSDL2main -lGL -lGLU -lglut -lGLEW -lstdc++fs
cflags = -Wall -Wextra -std=c++17

deps = ../input/*.cpp ../colorer/cam_calibration.h
cpps := $(shell find -name '*.cpp') ../colorer/cam_calibration.cpp
objects := $(cpps:%.cpp=%.o)

.PHONY : all
test : $(objects)
	$(cc) $(objects) $(deps) -o $@ $(libs) $(cflags)

.PHONY : clean
clean:
	-rm -f *.o test

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(cflags)
