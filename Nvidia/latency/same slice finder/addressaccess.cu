#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_SM 80
#define BLOCK_SIZE 32 // related with tid*N°
#define S_SIZE ((6*1024)*1024)/4 // must be smaller than L2 cache size
#define ITERATION 2

typedef unsigned int uint;

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

__global__ void kernel(unsigned int *a0, unsigned int *timing, unsigned int start_idx) {
    __shared__ unsigned int measured_latency[12288];
    unsigned int i,j,k,idx;
    unsigned int tid = threadIdx.x;
    unsigned int sm_id = get_smid();
    unsigned int start, latency;

    if(sm_id == 0) 
    {
        k = 0;
        __syncthreads();

        for (i = 0; i < ITERATION; i ++)
        {
            idx = start_idx*256;

            start = clock();

            for (j = 0; j < 2; j++){
                k += a0[idx];
                //printf("thread : %u\n", tid);
            }

            latency = clock() - start;

            a0[0] = k; // needed for forcing dependency

            if(tid == 0) {
                measured_latency[i] = latency;
                timing[i] = measured_latency[i];
            }
        }
        
    }

}
    
int main(int argc, char * argv[]) {
    unsigned int * h_arr;
    unsigned int * h_timing;
    unsigned int * d_a0;
    unsigned int * d_timing;
    int i,start_idx;
    int percentage;
    
    cudaSetDevice(0);

    percentage = atoi(argv[1]);
    start_idx = atoi(argv[2]);

    h_arr = (unsigned int *)malloc(sizeof(unsigned int) * S_SIZE);
    h_timing = (unsigned int *)malloc(sizeof(unsigned int) * ITERATION);

    cudaMalloc((void**)&d_a0, sizeof(unsigned int) * S_SIZE);
    cudaCheckError();

    cudaMalloc((void**)&d_timing, sizeof(unsigned int) * ITERATION);
    cudaCheckError();


    for (i = 0; i < S_SIZE; i++) {
        h_arr[i] = 10;
    }

    cudaMemcpy(d_a0, h_arr, sizeof(unsigned int) * S_SIZE, cudaMemcpyHostToDevice);
    cudaCheckError();

    int carveout = percentage; // prefer shared memory capacity 100% of maximum
    cudaFuncSetAttribute (kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaCheckError();

    kernel<<<NUM_SM, BLOCK_SIZE>>>(d_a0, d_timing,start_idx);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(h_timing, d_timing, sizeof(unsigned int) * ITERATION, cudaMemcpyDeviceToHost);
    cudaCheckError();

    for(int i = 0; i < ITERATION; i++)
        printf("measured latency is %u\n",h_timing[i]);

    free(h_arr);
    free(h_timing);
    cudaFree(d_a0); 
    cudaFree(d_timing);

    return 0;
}


