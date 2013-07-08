#include <iostream>
#include <exception>
#include <cstring>
#include <CL/cl.h>
#include <sys/times.h>
#include <cmath>
#include <stdio.h>
#include <unistd.h>


int main(int argc,char*argv[])
{
    size_t srcsize;
    size_t numThreads=256;
    const double PI25DT = 3.141592653589793238462643;
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint platforms, devices;
    clGetPlatformIDs(1, &platform, &platforms);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &devices);
    cl_context_properties properties[]= { CL_CONTEXT_PLATFORM, (cl_context_properties)platform,0};
    cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
    cl_command_queue cq = clCreateCommandQueue(context, device, 0, &error);
    clock_t clockStart, clockStop;
    tms tmsStart, tmsStop;
    char src[1024*1024];

    FILE *fil=fopen("7_opencl.cl","r");
    srcsize=fread(src, sizeof src, 1, fil);
    fclose(fil);

    clockStart = times(&tmsStart);
    std::cout<< "calculating pi with OpenCL\n";
    const char *srcptr[]= {src};
    cl_program prog=clCreateProgramWithSource(context, 1, srcptr, &srcsize, &error);
    clBuildProgram(prog, 0, NULL, "", NULL, NULL);
    cl_mem gpuBuf;
    gpuBuf=clCreateBuffer(context, CL_MEM_WRITE_ONLY, numThreads*sizeof(double), NULL, &error);
    cl_kernel kernelPi=clCreateKernel(prog, "pi", &error);
    clSetKernelArg(kernelPi, 0, sizeof(gpuBuf), &gpuBuf);
    size_t dimGrid=1;
    clEnqueueNDRangeKernel(cq, kernelPi, 1, NULL, &numThreads, &numThreads, 0, NULL, NULL);
    double  *cpuResults=new double[numThreads];
    clEnqueueReadBuffer(cq, gpuBuf, CL_FALSE, 0, numThreads*sizeof(double), cpuResults, 0, NULL, NULL);
    clFinish(cq);
    double pi=0.;
    for (int i=0; i<numThreads; i++)
    {
        pi+=cpuResults[i];
    }
    clockStop = times(&tmsStop);
    std::cout << "The value of PI is " << pi << " Error is " <<   fabs(pi - PI25DT) << std::endl;
    std::cout << "The time to calculate PI was " ;
    double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
    std::cout << secs << " seconds\n" << std::endl;
}
