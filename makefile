CFLAGS = -Wall -fopenmp -03
All:  1_sequential 2a_threads_mutex 2b_threads_no_mutex 3_openmp  4_tbb 5_mpi 6_cuda 7_opencl_stripped

1_sequential: 1_sequential.cpp
	g++ $^ -o $@ -O3
2a_threads_mutex: 2a_threads_mutex.cpp
	g++ $^ -o $@  -lpthread -O3

2b_threads_no_mutex: 2b_threads_no_mutex.cpp
	g++ $^ -o $@  -lpthread -O3

3_openmp: 3_openmp.cpp
	g++ $^ -o $@ -fopenmp -O3

4_tbb: 4_tbb.cpp
	g++ $^ -o $@ -ltbb -O3

5_mpi: 5_mpi.cpp
	mpiCC $^ -o $@

6_cuda: 6_cuda.cu
	nvcc -arch sm_20 $^ -o $@

7_opencl_stripped
	g++ $^ -o $@ -O3