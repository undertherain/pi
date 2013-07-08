#include <cstdio>
#include <omp.h>
#include "offload.h"

#define MIC_DEV 0

int main(int argc, char *argv[])
{
	#pragma offload target(mic:MIC_DEV)
	{
		#pragma omp parallel
		{
	    	printf ("num threads=%d\n", omp_get_num_threads());
		}
	}
    return 0;
}