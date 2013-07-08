#include <cstdio>
#include <omp.h>
#include "offload.h"
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <sys/times.h>


#define MIC_DEV 0

int main(int argc, char *argv[])
{
	const unsigned long numSteps=500000000;	
    double PI25DT = 3.141592653589793238462643;
    double pi=0;
    double sum=0.0;
    double x;
	double step;

    clock_t clockStart, clockStop;
    tms tmsStart, tmsStop;
    step = 1./static_cast<double>(numSteps);

	#pragma offload target(mic:MIC_DEV) inout(sum)  in(step) // first call to device might take some time
	{
		#pragma omp parallel
		{
			#pragma omp master
			{
	    		printf ("num threads=%d\n", omp_get_num_threads());
	    	}
		}
	}
    clockStart = times(&tmsStart);
	#pragma offload target(mic:MIC_DEV) inout(sum)  in(step)
	{
    	#pragma omp  parallel for private (x), reduction (+:sum)
   		for (int i=0; i<numSteps; i++)
   		{
       		x = (i + .5)*step;
        	sum = sum + 4.0/(1.+ x*x);
	    }
	}
    pi = sum*step;
    clockStop = times(&tmsStop);
    std::cout << "The value of PI is " << pi << " Error is " << fabs(pi - PI25DT) << std::endl;
    std::cout << "The time to calculate PI was " ;
    double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
    std::cout << secs << " seconds\n" << std::endl;
    return 0;
}