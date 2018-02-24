#Representation makefile

cc = g++
cflags = -Wall -Wextra -std=c++17

cpps := $(shell find -name '*.cpp')
objects := $(cpps:%.cpp=%.o)

.PHONY : all
all : $(objects)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(cflags)
