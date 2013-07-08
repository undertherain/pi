#include <iostream>
#include <exception>
#include <cstring>
#include <CL/cl.h>
#include <sys/times.h>
#include <cmath>
#include <stdio.h>
#include <stdexcept>
#include "7_clerrors.hpp"
#include <unistd.h>

int main(int argc, char * argv[]) {
	size_t cntThreads=256;
	const double PI25DT = 3.141592653589793238462643;
	cl_int error;
	cl_platform_id platform;
	cl_device_id device;
	cl_uint platforms, devices;
	clock_t clockStart, clockStop;
	tms tmsStart, tmsStop;
    clockStart = times(&tmsStart);

	std::cout<< "calculating pi with OpenCL\n";
	// Fetch the Platform and Device IDs; we only want one.
	error=clGetPlatformIDs(1, &platform, &platforms);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}
	error=clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &devices);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}
	cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
	// Note that nVidia's OpenCL requires the platform property
	cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}
	cl_command_queue cq = clCreateCommandQueue(context, device, 0, &error);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}

	char src[1024*1024];
	FILE *fileClSource=fopen("7_opencl.cl","r");
	size_t srcsize=fread(src, sizeof src, 1, fileClSource);
	fclose(fileClSource);

	const char *srcptr[]={src};
	// Submit the source code of the rot13 kernel to OpenCL
	cl_program prog=clCreateProgramWithSource(context, 1, srcptr, &srcsize, &error);
	// and compile it (after this we could extract the compiled version)
	error=clBuildProgram(prog, 0, NULL, "", NULL, NULL);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}

	// Allocate memory for the kernel to work with
	cl_mem gpuBuf;
	gpuBuf=clCreateBuffer(context, CL_MEM_WRITE_ONLY, cntThreads*sizeof(double), NULL, &error);

	// get a handle and map parameters for the kernel
	cl_kernel kernelPi=clCreateKernel(prog, "pi", &error);
	if (error)
	{
		std::cerr<<"cl error: "<<cl_error_str(error)<<std::endl;
		throw 1;
	}
	clSetKernelArg(kernelPi, 0, sizeof(gpuBuf), &gpuBuf);

	// Perform the operation
	size_t dimGrid=1;
	error=clEnqueueNDRangeKernel(cq, kernelPi, 1, NULL, &cntThreads, &cntThreads, 0, NULL, NULL);
	if (error)
	{
		std::cerr<<"cl error: "<<cl_error_str(error)<<std::endl;
		throw 1;
	}
	// Target buffer just so we show we got the data from OpenCL
	double  *cpuResults=new double[cntThreads];

	// Read the result back into buf2
	error=clEnqueueReadBuffer(cq, gpuBuf, CL_FALSE, 0, cntThreads*sizeof(double), cpuResults, 0, NULL, NULL);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}
	// Await completion of all the above
	error=clFinish(cq);
	if (error)
	{
		throw std::runtime_error(cl_error_str(error));
	}
	// Finally, output out happy message.
	double pi=0.;
	for (unsigned int i=0;i<cntThreads;i++)
	{
		pi+=cpuResults[i];
	}
	clockStop = times(&tmsStop);
	std::cout << "The value of PI is " << pi << " Error is " <<   fabs(pi - PI25DT) << std::endl;
	std::cout << "The time to calculate PI was " ;
	double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
	std::cout << secs << " seconds\n" << std::endl;
}
