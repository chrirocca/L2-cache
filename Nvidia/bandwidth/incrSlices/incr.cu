#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <sys/time.h>

#define BLOCK_SIZE 32 // related with tid*N°
#define ADDRESS_BLOCK 256
#define S_SIZE ((6*1024)*1024)/4 // must be smaller than L2 cache size
#define ITERATION 10000
#define MULTIPLIER 1

typedef unsigned int uint;

#define USECPSEC 1000000ULL

#define DIV_ROUND_CLOSEST(n, d) ((((n) < 0) == ((d) < 0)) ? (((n) + (d)/2)/(d)) : (((n) - (d)/2)/(d)))

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(0); \
    }                                                                 \
}

__device__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}

double dtime_usec(unsigned long long start=0){
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}


__global__ void kernel(unsigned int *a0, unsigned int *value, unsigned int MPnum){
    unsigned int k[BLOCK_SIZE*BLOCK_SIZE];
    unsigned int idx[BLOCK_SIZE*BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int warp = tid/32;
    unsigned int tx = tid%32;
    unsigned int sm_id = get_smid();
    unsigned int bid = blockIdx.y%MPnum;

    int temp[BLOCK_SIZE*MULTIPLIER*10];
    for (int i=0;i<BLOCK_SIZE*MULTIPLIER*MPnum;i++) temp[i] = value[i];
    __syncthreads();

        for (int i = 0; i < ITERATION; i ++){
            idx[tid] = warp*8 + temp[tx+BLOCK_SIZE*MULTIPLIER*bid]*ADDRESS_BLOCK;
            for (int j=0;j<2;j++){
                k[tid] += a0[idx[tid]];
            }
            a0[sm_id*ADDRESS_BLOCK] = k[tid];
        }
    }

    
    
int main(int argc, char * argv[]) {
    unsigned int * h_arr;
    unsigned int * d_a0;

    unsigned int * d_sel;

    int tx_max;
    int percentage;


    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    const int NUM_SM = device_prop.multiProcessorCount; // number of SMs

    cudaSetDevice(0);

    percentage = 100;
    tx_max = 32;
    unsigned int MPnum = atoi(argv[1]);



    h_arr = (unsigned int *)malloc(sizeof(unsigned int) * S_SIZE);


     unsigned int h_matrix[4*BLOCK_SIZE] = {
    0, 4, 11, 15, 16, 20, 27, 31, 33, 37, 42, 46, 49, 53, 58, 62, 64, 68, 75, 79, 80, 84, 91, 95, 97, 101, 106, 110, 113, 117, 122, 126,
    2, 6, 9, 13, 18, 22, 25, 29, 35, 39, 40, 44, 51, 55, 56, 60, 66, 70, 73, 77, 82, 86, 89, 93, 99, 103, 104, 108, 115, 119, 120, 124, 
    3, 7, 8, 12, 19, 23, 24, 28, 34, 38, 41, 45, 50, 54, 57, 61, 67, 71, 72, 76, 83, 87, 88, 92, 98, 102, 105, 109, 114, 118, 121, 125,
    1, 5, 10, 14, 17, 21, 26, 30, 32, 36, 43, 47, 48, 52, 59, 63, 65, 69, 74, 78, 81, 85, 90, 94, 96, 100, 107, 111, 112, 116, 123, 127
    };


    

    unsigned int *h_sel;
    h_sel = (unsigned int *)malloc(sizeof(unsigned int) * BLOCK_SIZE*MULTIPLIER*MPnum);

    cudaMalloc((void**)&d_a0, sizeof(unsigned int) * S_SIZE);
    cudaMalloc((void**)&d_sel, sizeof(unsigned int) * BLOCK_SIZE*MULTIPLIER*MPnum);
    cudaCheckError();



    for (int i = 0; i < S_SIZE; i++) {
        h_arr[i] = i;
    }

        for (int i = 0; i < BLOCK_SIZE*MULTIPLIER*MPnum; i++) {
            h_sel[i] = h_matrix[i];
        }

    cudaMemcpy(d_a0, h_arr, sizeof(unsigned int) * S_SIZE, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_sel, h_sel, sizeof(unsigned int) * BLOCK_SIZE*MULTIPLIER*MPnum, cudaMemcpyHostToDevice);
    cudaCheckError();

    int carveout = percentage; // prefer shared memory capaciwarp 100% of maximum
    cudaFuncSetAttribute (kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaCheckError();

    dim3 blockDim(BLOCK_SIZE*BLOCK_SIZE);
    dim3 GridDim(NUM_SM, MPnum, 1);

    double t1 = dtime_usec(0);
    kernel<<<GridDim, blockDim>>>(d_a0,d_sel, MPnum);
    cudaCheckError();
    cudaDeviceSynchronize();
    double t2 = dtime_usec(t1);
    cudaCheckError();

/*     measureL2BytesStart();
    kernel<<<NUM_SM, blockDim>>>(d_a0,d_sel,row_idx,tx_max, SMdiv);
    cudaDeviceSynchronize();
    auto metrics = measureL2BytesStop();
    L2_read.add(metrics[0]);
    L2_write.add(metrics[1]); 

    float L2_read_bw = L2_read.value() / time.minValue() / 1.0e3;
    float L2_write_bw = L2_write.value() / time.minValue() / 1.0e3; */

    float L2_read_bw = ITERATION*BLOCK_SIZE*(tx_max)*8*sizeof(unsigned int)*NUM_SM*MPnum / t2 / 1.0e3 ;

    printf("%f\n", L2_read_bw);


    //free(hrow_idx);
    free(h_arr);
    free(h_sel);
    //free(hSM_ids1);
    cudaFree(d_a0); 
    cudaFree(d_sel);
    //cudaFree(dSM_ids1);
    //cudaFree(drow_idx);

    return 0;
}


