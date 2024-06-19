#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_SM 120
#define BLOCK_SIZE 64 // related with tid*N°
#define S_SIZE ((8*1024)*1024)/16 // must be smaller than L2 cache size
#define ITERATION 1000000

typedef unsigned int uint;

#define hipCheckError() {                                          \
    hipError_t e=hipGetLastError();                                 \
    if(e!=hipSuccess) {                                              \
        printf("HIP failure %s:%d: '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));           \
        exit(0); \
    }                                                                 \
}


/*
   HW_ID Register bit structure
   WAVE_ID     3:0     Wave buffer slot number. 0-9.
   SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
   PIPE_ID     7:6     Pipeline from which the wave was dispatched.
   CU_ID       11:8    Compute Unit the wave is assigned to.
   SH_ID       12      Shader Array (within an SE) the wave is assigned to.
   SE_ID       15:13   Shader Engine the wave is assigned to.
   TG_ID       19:16   Thread-group ID
   VM_ID       23:20   Virtual Memory ID
   QUEUE_ID    26:24   Queue from which this wave was dispatched.
   STATE_ID    29:27   State ID (graphics only, not compute).
   ME_ID       31:30   Micro-engine ID.
 */

#define HW_ID               4

#define HW_ID_CU_ID_SIZE    4
#define HW_ID_CU_ID_OFFSET  8

#define HW_ID_SE_ID_SIZE    3
#define HW_ID_SE_ID_OFFSET  13

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
 */

/*
  __smid returns the wave's assigned Compute Unit and Shader Engine.
  The Compute Unit, CU_ID returned in bits 3:0, and Shader Engine, SE_ID in bits 5:4.
  Note: the results vary over time.
  SZ minus 1 since SIZE is 1-based.
*/

#define GETREG_IMMED(SZ,OFF,REG) (((SZ) << 11) | ((OFF) << 6) | (REG))

__device__ inline unsigned get_smid(void)
{
    unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE-1, HW_ID_CU_ID_OFFSET, HW_ID));
    unsigned se_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE-1, HW_ID_SE_ID_OFFSET, HW_ID));

    /* Each shader engine has 16 CU */
    return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}


__global__ void k(unsigned int *a0, unsigned int *a1, unsigned int start_idx, unsigned int sm_chosen) {
    unsigned int i;
    unsigned int sm_id = get_smid();
    unsigned int warp = threadIdx.x/64;
    unsigned int tx = threadIdx.x%64;
    unsigned int temp = warp%8;
    
    if(sm_id == sm_chosen) 
    {

        for (i = 0; i < ITERATION; i ++)
        {


            a0[(tx%8)*start_idx+temp*8]+= a1[0];


        }
        
    }

}
    
int main(int argc, char * argv[]) {
    unsigned int * h_arr;
    unsigned int * d_a0, * d_a1;
    int i,start_idx, sm_chosen;
    
    hipSetDevice(0);
    start_idx = atoi(argv[1])*64;
    sm_chosen = atoi(argv[2]);

    h_arr = (unsigned int *)malloc(sizeof(unsigned int) * S_SIZE);

    hipMalloc((void**)&d_a0, sizeof(unsigned int) * S_SIZE);
    hipCheckError();
    hipMalloc((void**)&d_a1, sizeof(unsigned int) * S_SIZE);
    hipCheckError();

    for (i = 0; i < S_SIZE; i++) {
        h_arr[i] = i;
    }

    hipMemcpy(d_a0, h_arr, sizeof(unsigned int) * S_SIZE, hipMemcpyHostToDevice);
    hipCheckError();
    hipMemcpy(d_a1, h_arr, sizeof(unsigned int) * S_SIZE, hipMemcpyHostToDevice);
    hipCheckError();

    k<<<NUM_SM, BLOCK_SIZE*16>>>(d_a0,d_a1,start_idx,sm_chosen);
    hipCheckError();
    hipDeviceSynchronize();
    hipCheckError();


    free(h_arr);
    hipFree(d_a0); 
    hipFree(d_a1); 

    return 0;
}


