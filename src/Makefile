cc = g++
cflags = -lSDL2 -lSDL2main -lGL -lGLU -lGLEW -Wall -Wextra -std=c++17
SOURCEDIR = .
BUILDDIR  = ../build

#deps = repr/input.o repr/point.o 

cpps = $(shell find -not -path "./test/*" -name '*.cpp')
#local_objects := $(addprefix $(BUILDDIR)/,$(SOURCES:%.c=%.o))
objects := $(cpps:%.cpp=%.o)


.PHONY : all
all : $(objects)
	$(cc) $(objects) -o main $(cflags)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(cflags)

.PHONY : test
test :
	cd test; make
	./test/test

.PHONY : clean
clean :
	-rm -f input/*.o repr/*.o test/*.o *.o main
