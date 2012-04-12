#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void pi(__global double* out)
{
   const uint idThread = get_local_id(0);
   const uint cntSteps=500000000;
   uint numprocs=get_global_size(0);
   const double local_num= (double)cntSteps / numprocs;
   double step = 1.0 / cntSteps;
   double sum=0;
   double x;
   int localmax = (idThread+1)*local_num;
   for (uint i = idThread*local_num; i < localmax; i ++)
   {
      x = step * (i + 0.5);
      sum = sum + 4.0 / (1.0 + x*x);
   }
   out[idThread]=sum*step;
}


