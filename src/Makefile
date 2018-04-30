cc = g++
gpucc = nvcc
gpuflags = -ccbin=clang++-3.8 -rdc=true -Xcompiler -fPIC -arch=sm_35
gpulibs = -lcudart -lcudadevrt
libs = -lSDL2 -lSDL2main -lGL -lGLU -lglut -lGLEW -lstdc++fs
cflags = -Wall -Wextra -std=c++17 -O2

SOURCEDIR = .
BUILDDIR  = ../build

cpps = $(shell find -not -path "./test/*" -not -path "./colorer/*" -name '*.cpp')
hpps = display/program.hpp
cus = $(shell find -not -path "./test/*" -name '*.cu')
objects := $(cpps:%.cpp=%.o)
cuobjects := $(cus:%.cu=%.o)

imgui_files = $(shell find -name ../3rd/imgui/*.cpp)
imgui_objs := $(cpps:%.cpp=%.o)


.PHONY : all
all : $(objects) $(hpps) $(cuobjects) $(imgui_objects)
	$(gpucc) -dlink -o link.o $(cuobjects) $(gpuflags) $(gpulibs)
	$(cc) $(objects) $(cuobjects) $(imgui_objects) $(hpps) link.o -o main $(libs) $(gpulibs) $(cflags)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(libs) $(cflags)

$(imgui_objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(libs) $(cflags)

$(cuobjects) : %.o : %.cu
	$(gpucc) -c $< -o $@ $(gpuflags) $(gpulibs)

.PHONY : test
test :
	cd test; make
	./test/test

.PHONY : clean
clean :
	-rm -f input/*.o input/*.o test/*.o display/*.o gpu/*.o *.o 3rd/imgui/*.o main

.PHONY: rebuild
rebuild : clean all
