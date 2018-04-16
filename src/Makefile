cc = g++
libs = -lSDL2 -lSDL2main -lopencv_core -lopencv_highgui -lGL -lGLU -lglut -lGLEW -lstdc++fs
cflags = -Wall -Wextra -std=c++17 -O2

SOURCEDIR = .
BUILDDIR  = ../build

cpps = $(shell find -not -path "./test/*" -name '*.cpp')
hpps = display/program.hpp
objects := $(cpps:%.cpp=%.o)




.PHONY : all
all : $(objects) $(hpps)
	$(cc) $(objects) $(hpps) -o main $(libs) $(cflags)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(libs) $(cflags)

.PHONY : test
test :
	cd test; make
	./test/test

.PHONY : clean
clean :
	-rm -f input/*.o util/*.o test/*.o display/*.o *.o main

.PHONY: rebuild
rebuild : clean all
