#include <iostream>
#include <cmath>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_stddef.h>
#include <sys/times.h>

class Sum
{
    double step;
public:
    float value;
    Sum(double _step):  value(0),step(_step) {}
    Sum(Sum &s, tbb::split)
    {
        value = 0;
    }
    void operator() (const tbb::blocked_range<size_t>& range )
    {
        int a=range.begin();
        int b=range.end();
        double x,partialSum;
        partialSum=value;
        for (int i=a; i<b; i++)
        {
            x = (i + .5)*step;
            partialSum += 4.0/(1.+ x*x);
        }
        value=partialSum;
    }
    void join (Sum &rhs)
    {
        value+=rhs.value;
    }
};

int main(int argc, char* argv[])
{
    size_t const cntSteps = 500000000;
    double PI25DT = 3.141592653589793238462643;
    std::cout<< "calculating pi on CPU with TBB\n";
    clock_t clockStart, clockStop;
    tms tmsStart, tmsStop;
    double step = 1./static_cast<double>(cntSteps);
    clockStart = times(&tmsStart);
    tbb::task_scheduler_init init;
    Sum  mysum(step) ;
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, cntSteps, 1000), mysum );
    double pi = mysum.value * step;
    clockStop = times(&tmsStop);
    double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
    std::cout << "The value of PI is " << pi << " Error is " << fabs(pi - PI25DT) << "\nThe time to calculate PI was " << secs << " seconds" << std::endl;
    return 0;
}

