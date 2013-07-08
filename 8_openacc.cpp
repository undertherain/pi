#include <iostream>
#include <iomanip>
#include <sys/times.h>
#include <cmath>

int main(int argc, char** argv)
{
    const unsigned long cntSteps=500000000;	/* # of rectangles */
    double step; 
    const double PI25DT = 3.141592653589793238462643; //reference Pi value
    double pi=0;
    double sum=0.0;
    double x;
    std::cout<< "calculating pi on CPU single-threaded\n";
    clock_t clockStart, clockStop;  
    tms tmsStart, tmsStop; 
    step = 1./static_cast<double>(cntSteps);
    clockStart = times(&tmsStart);
#pragma omp parallel for default(none) shared(sum) private(x)
    for (unsigned long i=0; i<cntSteps; i++)
    {
        x = (i + .5)*step;
        sum = sum + 4.0/(1.+ x*x);
    }
    pi = sum*step;
    clockStop = times(&tmsStop);
    std::cout << "The value of PI is " << pi << " Error is " << fabs(pi - PI25DT) << "\n";
    std::cout << "The time to calculate PI was " ;
    double secs= (clockStop - clockStart)/static_cast<double>(sysconf(_SC_CLK_TCK));
    std::cout << secs << " seconds\n";
    return 0;
}

