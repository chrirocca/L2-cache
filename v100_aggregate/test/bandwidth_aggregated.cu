#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
#include "metrics.cuh"
#include "MeasurementSeries.hpp"
#include "gpu-metrics/gpu-metrics.hpp"
#include <cstdlib>
#include <ctime>

#define USECPSEC 1000000ULL

// find largest power of 2
unsigned flp2(unsigned x) {
    x = x| (x>>1);
    x = x| (x>>2);
    x = x| (x>>4);
    x = x| (x>>8);
    x = x| (x>>16);
    return x - (x>>1);
}

float mean(float arr[], int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

// function to measure time in microseconds
unsigned long long dtime_usec(unsigned long long start=0){
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

using mt = double;


// kernel
__global__ void l2_bw(double *dsink, int L2_SIZE){
// block and thread index
unsigned int tid = threadIdx.x;
unsigned int bid = blockIdx.x;
unsigned int THREADS = blockDim.x;
unsigned int THREADS_NUM = blockDim.x*gridDim.x;
// a register to avoid compiler optimization
double sink = 0;
double *posArray = dsink;
// load data from l2 cache and accumulate,
// l2 cache is warmed up before the launch of this kernel.
for(unsigned int i = 0; i<L2_SIZE; i+=THREADS_NUM){
double* ptr = posArray+i;
// every warp loads all data in l2 cache
for(unsigned int j = 0; j<THREADS; j+=32){
unsigned int offset = (tid+j)%THREADS;
asm volatile ("{\t\n"
".reg .f64 data;\n\t"
"ld.global.cg.f64 data, [%1];\n\t"
"add.f64 %0, data, %0;\n\t"
"}" : "+d"(sink) : "l"(ptr+offset) : "memory"
);
}
}
// store the result
dsink[tid*256] = sink;
}


int main(int argc, char** argv){
    cudaSetDevice(2);

    initMeasureMetric();

    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 2);

    cudaFuncSetAttribute (l2_bw, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    MeasurementSeries L2_read;
    MeasurementSeries L2_write;

    int CTA = atoi(argv[1]); // number of thread blocks per SM;
    int WARP = atoi(argv[2]); // number of warps per thread block;
    int ITERATION = atoi(argv[3]); // number of iterations to run;

    float bw[ITERATION];
/*     float bw_1SM[ITERATION];
    float bw_1SM_calc[ITERATION]; */
    float l2read[ITERATION];
    float l2write[ITERATION];

    float bw_mean = 0;
    float l2read_mean = 0;
    float l2write_mean = 0;
/*     float bw_mean_1SM = 0;
    float bw_mean_1SM_calc = 0; */

    //const int nTPSM = prop.maxThreadsPerMultiProcessor; // maximum number of threads per SM
    //printf("maxThreadsPerMultiProcessor: %d\n", nTPSM);
    const int nSM = prop.multiProcessorCount; // number of SMs
    const unsigned l2size = prop.l2CacheSize; // L2 cache size

    unsigned sz = flp2(l2size); // size of data to transfer
    sz = sz/sizeof(mt);  // convert to number of elements

    int nTPB = 32; // block size
    int nBLK = nSM; // number of blocks
    const int loops = 100; // number of loops to run

    mt *d; // pointers for buffers on device and host

    cudaMalloc(&d, sz*sizeof(mt)); // allocate memory on device

std::cout << std::setw(13) << "         bandwidth      "   //
                    << std::setw(12) << "    L2_read     "  //
                    << std::setw(11) << "    L2_write    "     //
/*                     << std::setw(25) << "    Theoretical Bandwidth 1 SM    "     //
                    << std::setw(25) << "    Calculated Bandwidth 1 SM   "     // */
                    << std::setw(11) << "    CTA   "     //
                    << std::setw(11) << "    WARP  "     //
                    << std::setw(11) << "    ITERATION = " << ITERATION     //
                    << std::endl;

    for (int j = WARP; j <= WARP; j++){
        for (int i = 1; i <= CTA; i++){

            for (int h = 0; h < ITERATION; h++){

                l2_bw<<<nBLK*i, nTPB*j>>>(d, sz);  // warm-up
                cudaDeviceSynchronize(); // synchronize device    

                mt dt = dtime_usec(0); // get start time
                l2_bw<<<nBLK*i, nTPB*j>>>(d, sz); // run kernel
                
                cudaDeviceSynchronize(); // synchronize device
                dt = dtime_usec(dt); // get end time

                 measureL2BytesStart();
                l2_bw<<<nBLK*i, nTPB*j>>>(d, sz);
                cudaDeviceSynchronize();
                auto metrics = measureL2BytesStop();
                L2_read.add(metrics[0]);
                L2_write.add(metrics[1]); 

                l2read[h] = L2_read.median() / (dt * 1000);
                l2write[h] = L2_write.median() / (dt * 1000);


                bw[h] = (sz*2*sizeof(mt)*loops)/((float)dt*1000); // calculate bandwidth
/*                 bw_1SM[h] = bw[h] / nSM; //theoretical bandwidth per SM
                bw_1SM_calc[h] = (nTPB*j*i*sizeof(mt)*num_increments*loops*2)/((float)dt*1000); //theoretical bandwidth per SM */

            }

            bw_mean = mean(bw, ITERATION);
            l2read_mean = mean(l2read, ITERATION);
            l2write_mean = mean(l2write, ITERATION);
/*             bw_mean_1SM = mean(bw_1SM, ITERATION);
            bw_mean_1SM_calc = mean(bw_1SM_calc, ITERATION); */

            std::cout   << std::fixed << std::setprecision(2)
                        << std::setw(13) << bw_mean << " GB/s   "
                        << std::fixed << std::setprecision(2) << std::setw(10)
                        << l2read_mean << " GB/s "
                        << std::fixed << std::setprecision(2) << std::setw(10)
                        << l2write_mean << " GB/s "
/*                         << std::fixed << std::setprecision(2)
                        << std::setw(15) << bw_mean_1SM<< " GB/s   "
                        << std::fixed << std::setprecision(2)
                        << std::setw(30) << bw_mean_1SM_calc<< " GB/s   " */
                        << std::setw(15) << i << "  "
                        << std::setw(10) << j << "  "
                        << std::endl;            

        }
    }

}