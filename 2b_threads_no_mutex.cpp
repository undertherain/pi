#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <pthread.h>
#include <sys/times.h>
#include <unistd.h>

#define cntThreads 4

struct ArgsThread
{
    long long left,right;
    double step;
    double partialSum;
};

static void *worker(void *ptrArgs)
{
    ArgsThread * args = reinterpret_cast<ArgsThread *>(ptrArgs);
    double x;
    double sum=0.;
    double step=args->step;
    for (long long i=args->left; i<args->right; i++)
    {
        x = (i + .5)*step;
        sum = sum + 4.0/(1.+ x*x);
    }
    args->partialSum=sum*step;
    return NULL;
}

int main(int argc, char** argv)
{
    const unsigned long num_steps=500000000;
    const double PI25DT = 3.141592653589793238462643;
    pthread_t threads[cntThreads];
    ArgsThread arrArgsThread[cntThreads];
    std::cout<<"POSIX threads. number of threads = "<<cntThreads<<std::endl;
    clock_t clockStart, clockStop;
    tms tmsStart, tmsStop;
    clockStart = times(&tmsStart);
    double step = 1./(double)num_steps;
    long long cntStepsPerThread= num_steps / cntThreads;
    for (unsigned int idThread=0; idThread<cntThreads; idThread++)
    {
        arrArgsThread[idThread].left  = idThread*cntStepsPerThread;
        arrArgsThread[idThread].right = (idThread+1)*cntStepsPerThread;
        arrArgsThread[idThread].step = step;
        if (pthread_create(&threads[idThread], NULL, worker, &arrArgsThread[idThread]) != 0)
        {
            return EXIT_FAILURE;
        }
    }
    double pi=0.;
    for (unsigned int idThread=0; idThread<cntThreads; idThread++)
    {
        if (pthread_join(threads[idThread], NULL) != 0)
        {
            return EXIT_FAILURE;
        }
        pi +=arrArgsThread[idThread].partialSum;
    }
    clockStop = times(&tmsStop);
    std::cout << "The value of PI is " << pi << " Error is " << fabs(pi - PI25DT) << std::endl;
    std::cout << "The time to calculate PI was " ;
    double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
    std::cout << secs << " seconds\n" << std::endl;
    return 0;
}

