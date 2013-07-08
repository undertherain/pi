#include <iostream>
#include <cmath>
#include <mpi.h>
#include <sys/times.h>
#include <unistd.h>

int main(int argc, char* argv[])
{
    int idCurrentThread, cntThreads;
    int lenNameProcessor;
    char nameProcessor[MPI_MAX_PROCESSOR_NAME];
    double localPi, step, sum, x;
    double PI25DT = 3.141592653589793238462643;
    long cntSteps = 500000000;

    clock_t clockStart, clockStop;
    tms tmsStart, tmsStop;
    MPI::Init(argc,argv);
    cntThreads = MPI::COMM_WORLD.Get_size();
    idCurrentThread = MPI::COMM_WORLD.Get_rank();
    MPI::Get_processor_name(nameProcessor,lenNameProcessor);
    std::cout << "Process " << idCurrentThread << " of " << cntThreads << " is on " <<nameProcessor << std::endl;
    if (idCurrentThread == 0)
        clockStart = times(&tmsStart);
    step = 1./static_cast<double>(cntSteps);
    sum = 0.0;
    long cntStepsPerThread= cntSteps / cntThreads;
    long limitRightCurrentThread = (idCurrentThread+1)*cntStepsPerThread;
//  std::cout << "limitLeftCurrentThread" << idCurrentThread*cntStepsPerThread << std::endl;
//  std::cout << "limitRightCurrentThread" << limitRightCurrentThread << std::endl;
    for (long i = idCurrentThread*cntStepsPerThread; i < limitRightCurrentThread; i ++)
    {
        x = step * (i + 0.5);
        sum = sum + 4.0 / (1.0 + x*x);
    }
    localPi = step * sum;
    double pi=0.;
    MPI::COMM_WORLD.Reduce(&localPi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0);
    if (idCurrentThread == 0)
    {
        clockStop = times(&tmsStop);
        std::cout << "The value of PI is " << pi << " Error is " <<   fabs(pi - PI25DT) << std::endl;
        std::cout << "The time to calculate PI was " ;
        double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
        std::cout << secs << " seconds\n" << std::endl;
    }
    MPI::Finalize();
    return 0;
}

