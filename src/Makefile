cc = g++
libs = -lSDL2 -lSDL2main -lGL -lGLU -lglut -lGLEW -lstdc++fs
cflags = -Wall -Wextra -std=c++17

SOURCEDIR = .
BUILDDIR  = ../build

cpps = $(shell find -not -path "./test/*" -name '*.cpp')
objects := $(cpps:%.cpp=%.o)


.PHONY : all
all : $(objects)
	$(cc) $(objects) -o main $(libs) $(cflags)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(libs) $(cflags)

.PHONY : test
test :
	cd test; make
	./test/test

.PHONY : clean
clean :
	-rm -f input/*.o util/*.o test/*.o display/*.o *.o main
