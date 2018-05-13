#Test makefile

cc = g++
libs = -lSDL2 -lSDL2main -lGL -lGLU -lglut -lGLEW -lstdc++fs
cflags = -Wall -Wextra -std=c++17

gpucc = nvcc
gpuflags = -ccbin=clang++-3.8 -rdc=true -Xcompiler -fPIC -arch=sm_35
gpulibs = -lcudart -lcudadevrt
gpu_deps = ../planes/gpu_math.o

deps = ../input/*.cpp ../colorer/cam_calibration.h
cpps := $(shell find -name '*.cpp') ../colorer/cam_calibration.cpp
objects := $(cpps:%.cpp=%.o)

.PHONY : all
test : $(objects)
	$(cc) $(objects) $(deps) -o $@ $(libs) $(cflags)

.PHONY : gpu
gpu : $(gpu_deps)
	$(gpucc) -c -o gpu_math_test.o $(gpudeps) gpu_math_test.cu $(gpuflags) $(gpulibs)
	$(gpucc) -dlink -o link.o $(gpu_deps) gpu_math_test.o $(gpuflags) $(gpulibs)
	$(cc) -o gpu_test link.o gpu_math_test.o $(gpu_deps) $(cflags) $(gpulibs) -std=c++11

.PHONY : clean
clean:
	-rm -f *.o test

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(cflags)
