#include <stdlib.h>
#include <stdio.h>

#define ZCOPY_THREADS  32
#define ZCOPY_DEFLEN   30000000
#define ZCOPY_ITER     10           // as in STREAM benchmark
#define NUM_SM         108           // number of SMs on device

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() * 1.0e-3;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}
#else
#error unsupported platform
#endif

__global__ void zcopy (const double2 * __restrict__ src, 
                       double2 * __restrict__ dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = src[i];
    }
}    

struct zcopyOpts {
    int len;
};

static int processArgs (int argc, char *argv[], struct zcopyOpts *opts)
{
    int error = 0;
    memset (opts, 0, sizeof(*opts));
    while (argc) {
        if (*argv[0] == '-') {
            switch (*(argv[0]+1)) {
            case 'n':
                opts->len = atol(argv[0]+2);
                break;
            default:
                fprintf (stderr, "Unknown switch '%c%s'\n", '-', argv[0]+1);
                error++;
                break;
            }
        }
        argc--;
        argv++;
    }
    return error;
}

int main (int argc, char *argv[])
{
    double start, stop, elapsed, mintime;
    double2 *d_a, *d_b;
    int errors;
    struct zcopyOpts opts;

    int CTA = atoi(argv[1]);

    errors = processArgs (argc, argv, &opts);
    if (errors) {
        return EXIT_FAILURE;
    }
    opts.len = (opts.len) ? opts.len : ZCOPY_DEFLEN;

    {
        size_t stacksize;
        CUDA_SAFE_CALL (cudaDeviceGetLimit (&stacksize, cudaLimitStackSize));
        //printf ("stacksize = %lu\n", stacksize);
    }


    /* Allocate memory on device */
    CUDA_SAFE_CALL (cudaMalloc((void**)&d_a, sizeof(d_a[0]) * opts.len));
    CUDA_SAFE_CALL (cudaMalloc((void**)&d_b, sizeof(d_b[0]) * opts.len));
    
    /* Initialize device memory */
    CUDA_SAFE_CALL (cudaMemset(d_a, 0x00, sizeof(d_a[0]) * opts.len)); // zero
    CUDA_SAFE_CALL (cudaMemset(d_b, 0xff, sizeof(d_b[0]) * opts.len)); // NaN

    /* Compute execution configuration */
    dim3 dimBlock(ZCOPY_THREADS);
    //int threadBlocks = (opts.len + (dimBlock.x - 1)) / dimBlock.x;
    int threadBlocks = NUM_SM*CTA;
    dim3 dimGrid(threadBlocks);
    
    //printf ("zcopy: operating on vectors of %d double2s (= %.3f MB)\n", 
            //opts.len, (double)sizeof(d_a[0]) * opts.len /(1024*1024));
    //printf ("zcopy: using %d threads per block, %d blocks\n", 
            //dimBlock.x, dimGrid.x);

    mintime = fabs(log(0.0));
    for (int k = 0; k < ZCOPY_ITER; k++) {
        start = second();
        zcopy<<<dimGrid,dimBlock>>>(d_a, d_b, opts.len);
        CHECK_LAUNCH_ERROR();
        stop = second();
        elapsed = stop - start;
        if (elapsed < mintime) mintime = elapsed;
    }
    //printf ("zcopy: mintime = %.3f msec  throughput = %.2f GB/sec\n",
            //1.0e3 * mintime, (2.0e-9 * sizeof(d_a[0]) * opts.len) / mintime);
    printf("%.2f\n", (2.0e-9 * sizeof(d_a[0]) * opts.len) / mintime);

    CUDA_SAFE_CALL (cudaFree(d_a));
    CUDA_SAFE_CALL (cudaFree(d_b));

    return EXIT_SUCCESS;
}