.PHONY: all
all:
	nvcc -ccbin=clang++-3.8 -arch=sm_35 -c gpu_math.cu -rdc=true -Xcompiler -fPIC
	nvcc -ccbin=clang++-3.8 -arch=sm_35 -c gpu_ransac.cu -rdc=true -Xcompiler -fPIC
	nvcc -ccbin=clang++-3.8 -arch=sm_35 -dlink -o link.o gpu_math.o gpu_ransac.o -lcudart -lcudadevrt -Xcompiler -fPIC
	g++ cpu_ransac_prep.cpp cpu_math.cpp ../input/point.cpp ../input/input.cpp ../input/directory.cpp -lstdc++fs -lopencv_core -Wall -Wextra gpu_math.o gpu_ransac.o link.o -lcudart -lcudadevrt

.PHONY: clean
clean:
	-rm -f *.o *.out

.PHONY: re
re: clean all
