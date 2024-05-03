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

__device__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
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
__global__ void k(mt *d, mt *d2, int len, int lps, unsigned int *SM_ids, unsigned int GPC_size, unsigned int * timing){

    unsigned int tid = threadIdx.x;
    int sm_id = get_smid();
    clock_t start, end;
    volatile unsigned int temp1, temp2;

    if (sm_id == SM_ids[0]){
            //__syncthreads();
            start = clock64();
            for (int l = 0; l < lps; l++){
                for (int i = threadIdx.x+blockDim.x*0; i<len; i+=14*blockDim.x){
                //d[i] = tid + l + sm_id;
                //d2[i] = tid + l + sm_id;
                temp1 += (d[i]+d2[i]);
                }
            }
            //__syncthreads();
            end = clock64();
            timing[tid] = (unsigned int)(end - start);
            d[0] = temp1;
        }
    
    for (int j = 1; j < GPC_size; j++){
        if (sm_id == SM_ids[j]){
            //__syncthreads();
            for (int l = 0; l < lps; l++){
                for (int i = threadIdx.x+blockDim.x*j; i<len; i+=14*blockDim.x){
                //d[i] = tid + l + sm_id;
                //d2[i] = tid + l + sm_id;
                temp1 += (d[i]+d2[i]);
                }
            }
            //__syncthreads();
            d[0] = temp1;
        }
    }


}

__global__ void k1(mt *d, mt *d2, int len){

                for (int i = threadIdx.x+blockDim.x*blockIdx.x; i<len; i+=gridDim.x*blockDim.x){
                d[i] = __ldcg(d2+i);
                }

}

int main(int argc, char** argv){
    cudaSetDevice(0);

    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cudaFuncSetAttribute (k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    unsigned int * h_timing;
	unsigned int * d_timing;

    int CTA = atoi(argv[1]); // number of thread blocks per SM;
    int WARP = atoi(argv[2]); // number of warps per thread block;
    int ITERATION = atoi(argv[3]); // number of iterations to run;

    float bw[ITERATION];
    float timing[ITERATION];

    unsigned int GPC, GPC_size, SM_max, *hSM_ids, *dSM_ids;
    GPC = atoi(argv[4]);
    SM_max = atoi(argv[5]);

    switch(GPC){
/*         case 0:
        GPC_size = 14;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 0; hSM_ids[1]=1; hSM_ids[2]=12; hSM_ids[3]=13; hSM_ids[4]=24; hSM_ids[5]=25; hSM_ids[6]=36; hSM_ids[7]=37; hSM_ids[8]=48; hSM_ids[9]=49; hSM_ids[10]=60; hSM_ids[11]=61; hSM_ids[12]=70; hSM_ids[13]=71;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);     
        break;
        case 1:
        GPC_size = 14;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 2; hSM_ids[1]=3; hSM_ids[2]=14; hSM_ids[3]=15; hSM_ids[4]=26; hSM_ids[5]=27; hSM_ids[6]=38; hSM_ids[7]=39; hSM_ids[8]=50; hSM_ids[9]=51; hSM_ids[10]=62; hSM_ids[11]=63; hSM_ids[12]=72; hSM_ids[13]=73;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);
        break; */
        case 0:
        GPC_size = 14;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 0; hSM_ids[7]=1; hSM_ids[1]=12; hSM_ids[8]=13; hSM_ids[2]=24; hSM_ids[9]=25; hSM_ids[3]=36; hSM_ids[10]=37; hSM_ids[4]=48; hSM_ids[11]=49; hSM_ids[5]=60; hSM_ids[12]=61; hSM_ids[6]=70; hSM_ids[13]=71;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);     
        break;
        case 1:
        GPC_size = 14;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 2; hSM_ids[7]=3; hSM_ids[1]=14; hSM_ids[8]=15; hSM_ids[2]=26; hSM_ids[9]=27; hSM_ids[3]=38; hSM_ids[10]=39; hSM_ids[4]=50; hSM_ids[11]=51; hSM_ids[5]=62; hSM_ids[12]=63; hSM_ids[6]=72; hSM_ids[13]=73;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);
        break;
        case 2:
        GPC_size = 14;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size); 
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size); 
        hSM_ids[0] = 4; hSM_ids[1]=5; hSM_ids[2]=16; hSM_ids[3]=17; hSM_ids[4]=28; hSM_ids[5]=29; hSM_ids[6]=40; hSM_ids[7]=41; hSM_ids[8]=52; hSM_ids[9]=53; hSM_ids[10]=64; hSM_ids[11]=65; hSM_ids[12]=74; hSM_ids[13]=75;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);
        break;
        case 3:
        GPC_size = 14;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 6; hSM_ids[1]=7; hSM_ids[2]=18; hSM_ids[3]=19; hSM_ids[4]=30; hSM_ids[5]=31; hSM_ids[6]=42; hSM_ids[7]=43; hSM_ids[8]=54; hSM_ids[9]=55; hSM_ids[10]=66; hSM_ids[11]=67; hSM_ids[12]=76; hSM_ids[13]=77;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);
        break;
        case 4:
        GPC_size = 12;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 8; hSM_ids[1]=9; hSM_ids[2]=20; hSM_ids[3]=21; hSM_ids[4]=32; hSM_ids[5]=33; hSM_ids[6]=44; hSM_ids[7]=45; hSM_ids[8]=56; hSM_ids[9]=57; hSM_ids[10]=68; hSM_ids[11]=69;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);
        break;
        case 5:
        GPC_size = 12;
        hSM_ids = (unsigned int *)malloc(sizeof(unsigned int) * GPC_size);
        cudaMalloc((void**)&dSM_ids, sizeof(unsigned int) * GPC_size);
        hSM_ids[0] = 10; hSM_ids[1]=11; hSM_ids[2]=22; hSM_ids[3]=23; hSM_ids[4]=34; hSM_ids[5]=35; hSM_ids[6]=46; hSM_ids[7]=47; hSM_ids[8]=58; hSM_ids[9]=59; hSM_ids[10]=78; hSM_ids[11]=79;
        cudaMemcpy(dSM_ids, hSM_ids, sizeof(unsigned int) * GPC_size, cudaMemcpyHostToDevice);
        break;
    }

    if (SM_max < GPC_size){
        GPC_size = SM_max;
    }
    else if (SM_max >= GPC_size){
        GPC_size = GPC_size;
    }

    float bw_mean = 0;
    float timing_mean = 0;

    const int nSM = prop.multiProcessorCount; // number of SMs
    const unsigned l2size = prop.l2CacheSize; // L2 cache size

    unsigned sz = flp2(l2size)/2; // size of data to transfer
    sz = sz/sizeof(mt);  // convert to number of elements

    int nTPB = 32; // block size
    int nBLK = nSM; // number of blocks
    const int loops = 1; // number of loops to run

    mt *d, *d2; // pointers for buffers on device and host

	h_timing = (unsigned int *)malloc(sizeof(unsigned int) * nTPB * WARP);
    cudaMalloc((void**)&d_timing, sizeof(unsigned int) * nTPB * WARP);

    cudaMalloc(&d, sz*sizeof(mt)); // allocate memory on device
    cudaMalloc(&d2, sz*sizeof(mt)); // allocate memory on device

std::cout << std::setw(13) << "         bandwidth      "   //
                    << std::setw(13) << "    cycles     "   //
                    << std::setw(20) << "    CTA   "     //
                    << std::setw(11) << "    WARP  "     //
                    << std::setw(11) << "    ITERATION = " << ITERATION     //
                    << std::endl;

    for (int j = WARP; j <= WARP; j++){
        for (int i = 1; i <= CTA; i++){

            for (int h = 0; h < ITERATION; h++){

                k1<<<nBLK*i, nTPB*j>>>(d, d2, sz);  // warm-up
                cudaDeviceSynchronize(); // synchronize device    

                mt dt = dtime_usec(0); // get start time
                k<<<nBLK*i, nTPB*j>>>(d, d2, sz, loops, dSM_ids, GPC_size, d_timing); // run kernel
                
                cudaDeviceSynchronize(); // synchronize device
                dt = dtime_usec(dt); // get end time

	            cudaMemcpy(h_timing, d_timing, sizeof(unsigned int) *  nTPB * WARP, cudaMemcpyDeviceToHost);

                timing[h] = h_timing[0];

                bw[h] = (sz*2*sizeof(mt)*loops*GPC_size)/((float)(dt)*1000*14); // calculate bandwidth

            }

            bw_mean = mean(bw, ITERATION);
            timing_mean = mean(timing, ITERATION);

            std::cout   << std::fixed << std::setprecision(1)
                        << std::setw(13) << bw_mean << " GB/s   "
                        << std::fixed << std::setprecision(1)
                        << std::setw(13) << timing_mean << " cycles   "
                        << std::setw(10) << i << "  "
                        << std::setw(10) << j << "  "
                        << std::endl;            

        }
    }

}