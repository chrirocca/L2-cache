#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
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

void init(mt *d, int len) {
    std::srand(std::time(nullptr)); // seed the random number generator

    for (int i = 0; i < len; i++) {
        d[i] = std::rand();
    }
}

// kernel
__global__ void k(mt *d, int len, int lps){
    volatile __shared__ int s[32*32];

  for (int l = 0; l < lps; l++){
        for (int i = threadIdx.x+blockDim.x*blockIdx.x; i<len; i+=gridDim.x*blockDim.x){
        s[i%(32*32)] = d[i];
        }
    }
}

int main(int argc, char** argv){
    cudaSetDevice(0);

    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cudaFuncSetAttribute (k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int CTA = atoi(argv[1]); // number of thread blocks per SM;
    int WARP = atoi(argv[2]); // number of warps per thread block;
    int ITERATION = atoi(argv[3]); // number of iterations to run;

    float bw[ITERATION];

    float bw_mean = 0;

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
                    << std::setw(11) << "    CTA   "     //
                    << std::setw(11) << "    WARP  "     //
                    << std::setw(11) << "    ITERATION = " << ITERATION     //
                    << std::endl;

    for (int j = WARP; j <= WARP; j++){
        for (int i = 1; i <= CTA; i++){

            for (int h = 0; h < ITERATION; h++){

                k<<<nBLK*i, nTPB*j>>>(d, sz, 1);  // warm-up
                cudaDeviceSynchronize(); // synchronize device    

                mt dt = dtime_usec(0); // get start time
                k<<<nBLK*i, nTPB*j>>>(d, sz, loops); // run kernel
                
                cudaDeviceSynchronize(); // synchronize device
                dt = dtime_usec(dt); // get end time

                bw[h] = (sz*sizeof(mt)*loops)/((float)dt*1000); // calculate bandwidth

            }

            bw_mean = mean(bw, ITERATION);

            std::cout   << std::fixed << std::setprecision(2)
                        << std::setw(13) << bw_mean << " GB/s   "
                        << std::setw(15) << i << "  "
                        << std::setw(10) << j << "  "
                        << std::endl;            

        }
    }

}