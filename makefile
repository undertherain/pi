CFLAGS = -Wall -fopenmp -03
CPPFLAGS :=  -xhost  -openmp
#CPPFLAGS :=  -mmic  -openmp

All:  1_sequential 2a_threads_mutex 2b_threads_no_mutex 3_openmp  4_tbb 5_mpi 6_cuda 7_opencl_stripped 8_openacc

1_sequential: 1_sequential.cpp
	g++ $^ -o $@ -O3

2a_threads_mutex: 2a_threads_mutex.cpp
	g++ $^ -o $@  -lpthread -O3

2b_threads_no_mutex: 2b_threads_no_mutex.cpp
	g++ $^ -o $@  -lpthread -O3

3_openmp: 3_openmp.cpp
	g++ $^ -o $@ -fopenmp -O3

3_openmp_offload: 3_openmp_offload.cpp
	icc $(CPPFLAGS) -o $@ $^

4_tbb: 4_tbb.cpp
	g++ $^ -o $@ -ltbb -O3

5_mpi: 5_mpi.cpp
	mpiCC $^ -o $@

6_cuda: 6_cuda.cu
	export GCC_CUDA=1; nvcc -arch sm_20 $^ -o $@

7_opencl_stripped: 7_opencl_stripped.cpp
	g++ $^ -o $@ -O3 -lOpenCL

8_openacc: 8_openacc.cpp
	g++ $^ -o $@ -O3 -lOpenCL



run:
	#source /opt/intel/composerxe/bin/compilervars.sh intel64
	export LD_LIBRARY_PARH=$(LD_LIBRARY_PARH):/opt/intel/composer_xe_2013.0.079/compiler/lib/intel64/
#sudo micctrl -s