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

using mt = unsigned long long;

void init(mt *d, mt *d2, int len) {
    std::srand(std::time(nullptr)); // seed the random number generator

    for (int i = 0; i < len; i++) {
        d[i] = std::rand();
        d2[i] = std::rand();
    }
}

// kernel
__global__ void k(mt *d, mt *d2, int len, int lps){

  for (int l = 0; l < lps; l++){
        for (int i = threadIdx.x+blockDim.x*blockIdx.x; i<len; i+=gridDim.x*blockDim.x){
        d[i] = __ldcg(d2+i);
        }
    }
}

int main(int argc, char** argv){
    cudaSetDevice(2);

    initMeasureMetric();

    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 2);

    cudaFuncSetAttribute (k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

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

    unsigned sz = flp2(l2size)/2; // size of data to transfer
    sz = sz/sizeof(mt);  // convert to number of elements

    int nTPB = 32; // block size
    int nBLK = nSM; // number of blocks
    const int loops = 100; // number of loops to run

    mt *d, *d2; // pointers for buffers on device and host

    cudaMalloc(&d, sz*sizeof(mt)); // allocate memory on device
    cudaMalloc(&d2, sz*sizeof(mt)); // allocate memory on device

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

                k<<<nBLK*i, nTPB*j>>>(d, d2, sz, 1);  // warm-up
                cudaDeviceSynchronize(); // synchronize device    

                mt dt = dtime_usec(0); // get start time
                k<<<nBLK*i, nTPB*j>>>(d, d2, sz, loops); // run kernel
                
                cudaDeviceSynchronize(); // synchronize device
                dt = dtime_usec(dt); // get end time

                 measureL2BytesStart();
                k<<<nBLK*i, nTPB*j>>>(d, d2, sz, loops);
                cudaDeviceSynchronize();
                auto metrics = measureL2BytesStop();
                L2_read.add(metrics[0]);
                L2_write.add(metrics[1]); 

                l2read[h] = L2_read.median() / (dt * 1000);
                l2write[h] = L2_write.median() / (dt * 1000);

/*                 int initial_i = 0;  // In your kernel, i starts from threadIdx.x + blockDim.x * blockIdx.x, which is 0 when blockIdx.x and threadIdx.x are 0
                int increment = nBLK *i * nTPB*j;
                int num_increments = (sz - initial_i) / increment; */

                // If len is not a multiple of increment, add 1 to num_increments
/*                 if ((sz - initial_i) % increment != 0) {
                    num_increments++;
                } */

                //printf("num_increments: %d\n", num_increments);

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