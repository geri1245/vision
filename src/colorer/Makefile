cc = g++
cflags = -std=c++17 -O2 -Wall -Wextra
libs = -lstdc++fs -lopencv_core -lopencv_highgui -lopencv_imgcodecs

cpps = $(shell find -name '*.cpp')
cpps += ../input/input.cpp ../input/directory.cpp ../input/point.cpp
objects := $(cpps:%.cpp=%.o)

.PHONY : all
all : $(objects)
	$(cc) $(objects) -o colorer $(cflags) $(libs)

$(objects) : %.o : %.cpp
	$(cc) -c $< -o $@ $(libs) $(cflags)

.PHONY : clean
clean :
	-rm -f $(objects) colorer

.PHONY: rebuild
rebuild : clean all
