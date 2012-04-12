#include <iostream>
#include <sys/times.h>

__global__ void calcInterval (double * data, const long cntSteps, int size, const double step)
{
    double x;
    double sum=0.0;
    int idThread=blockDim.x * blockIdx.x + threadIdx.x;;
    long local_num= cntSteps / size;
    long localmax = (idThread+1)*local_num;
    if (idThread==size-1) localmax=cntSteps;
    for (long i = idThread*local_num; i < localmax; i ++)
    {
        x = (i + .5)*step;
        sum = sum + 4.0/(1.+ x*x);
    }
    data[idThread]=sum;
}

int main(int argc, char** argv)
{
    const unsigned long cntSteps=500000000;			/* default # of rectangles */
    double step = 1./static_cast<double>(cntSteps);
    const double PI25DT = 3.141592653589793238462643;
    double pi=0.;
    const int cntThreads=256;
    const int cntBlocks=256;
    const long cntThreadsTotal=cntThreads*cntBlocks;
    std::cout << "\ncomputing on GPU (" << cntThreadsTotal << ") threads "  << std::endl;
    double * cpuValues = new double[cntThreadsTotal];
    double * gpuValues = NULL;
    clock_t clockStart, clockStop;
    tms tmsStart, tmsStop;
    clockStart = times(&tmsStart);
    cudaMalloc((void**) &gpuValues,cntThreadsTotal*sizeof(double));
    calcInterval<<<cntBlocks,cntThreads>>>(gpuValues,cntSteps,cntThreadsTotal,step);
    cudaMemcpy (cpuValues, gpuValues, cntThreadsTotal * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree   (gpuValues);
    for (long i=0; i<cntThreadsTotal; i++)
    {
        pi+=cpuValues[i]*step;
    }
    delete[] cpuValues;
    clockStop = times(&tmsStop);
    std::cout << "The value of PI is " << pi << " Error is " <<   fabs(pi - PI25DT) << std::endl;
    std::cout << "The time to calculate PI was " ;
    double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
    std::cout << secs << " seconds\n" << std::endl;
    return 0;
}

