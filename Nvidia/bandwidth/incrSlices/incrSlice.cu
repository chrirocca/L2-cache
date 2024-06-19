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

    __shared__ int temp[BLOCK_SIZE*MULTIPLIER*32];
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


     unsigned int h_matrix[32*BLOCK_SIZE] = {
    42, 58, 110, 126, 171, 187, 239, 255, 264, 280, 332, 348, 393, 409, 461, 477, 544, 560, 612, 628, 673, 689, 741, 757, 770, 786, 838, 854, 899, 915, 967, 983,
8, 24, 76, 92, 137, 153, 205, 221, 298, 314, 366, 382, 427, 443, 495, 511, 514, 530, 582, 598, 643, 659, 711, 727, 800, 816, 868, 884, 929, 945, 997, 1013,
0, 16, 68, 84, 129, 145, 197, 213, 290, 306, 358, 374, 419, 435, 487, 503, 522, 538, 590, 606, 651, 667, 719, 735, 808, 824, 876, 892, 937, 953, 1005, 1021,
34, 50, 102, 118, 163, 179, 231, 247, 256, 272, 324, 340, 385, 401, 453, 469, 552, 568, 620, 636, 681, 697, 749, 765, 778, 794, 846, 862, 907, 923, 975, 991,
46, 62, 106, 122, 175, 191, 235, 251, 268, 284, 328, 344, 397, 413, 457, 473, 548, 564, 608, 624, 677, 693, 737, 753, 774, 790, 834, 850, 903, 919, 963, 979,
12, 28, 72, 88, 141, 157, 201, 217, 302, 318, 362, 378, 431, 447, 491, 507, 518, 534, 578, 594, 647, 663, 707, 723, 804, 820, 864, 880, 933, 949, 993, 1009,
4, 20, 64, 80, 133, 149, 193, 209, 294, 310, 354, 370, 423, 439, 483, 499, 526, 542, 586, 602, 655, 671, 715, 731, 812, 828, 872, 888, 941, 957, 1001, 1017,
38, 54, 98, 114, 167, 183, 227, 243, 260, 276, 320, 336, 389, 405, 449, 465, 556, 572, 616, 632, 685, 701, 745, 761, 782, 798, 842, 858, 911, 927, 971, 987,
43, 59, 111, 127, 170, 186, 238, 254, 265, 281, 333, 349, 392, 408, 460, 476, 545, 561, 613, 629, 672, 688, 740, 756, 771, 787, 839, 855, 898, 914, 966, 982,
9, 25, 77, 93, 136, 152, 204, 220, 299, 315, 367, 383, 426, 442, 494, 510, 515, 531, 583, 599, 642, 658, 710, 726, 801, 817, 869, 885, 928, 944, 996, 1012,
1, 17, 69, 85, 128, 144, 196, 212, 291, 307, 359, 375, 418, 434, 486, 502, 523, 539, 591, 607, 650, 666, 718, 734, 809, 825, 877, 893, 936, 952, 1004, 1020,
35, 51, 103, 119, 162, 178, 230, 246, 257, 273, 325, 341, 384, 400, 452, 468, 553, 569, 621, 637, 680, 696, 748, 764, 779, 795, 847, 863, 906, 922, 974, 990,
47, 63, 107, 123, 174, 190, 234, 250, 269, 285, 329, 345, 396, 412, 456, 472, 549, 565, 609, 625, 676, 692, 736, 752, 775, 791, 835, 851, 902, 918, 962, 978,
13, 29, 73, 89, 140, 156, 200, 216, 303, 319, 363, 379, 430, 446, 490, 506, 519, 535, 579, 595, 646, 662, 706, 722, 805, 821, 865, 881, 932, 948, 992, 1008,
5, 21, 65, 81, 132, 148, 192, 208, 295, 311, 355, 371, 422, 438, 482, 498, 527, 543, 587, 603, 654, 670, 714, 730, 813, 829, 873, 889, 940, 956, 1000, 1016,
39, 55, 99, 115, 166, 182, 226, 242, 261, 277, 321, 337, 388, 404, 448, 464, 557, 573, 617, 633, 684, 700, 744, 760, 783, 799, 843, 859, 910, 926, 970, 986,
10, 26, 78, 94, 139, 155, 207, 223, 296, 312, 364, 380, 425, 441, 493, 509, 512, 528, 580, 596, 641, 657, 709, 725, 802, 818, 870, 886, 931, 947, 999, 1015,
40, 56, 108, 124, 169, 185, 237, 253, 266, 282, 334, 350, 395, 411, 463, 479, 546, 562, 614, 630, 675, 691, 743, 759, 768, 784, 836, 852, 897, 913, 965, 981,
32, 48, 100, 116, 161, 177, 229, 245, 258, 274, 326, 342, 387, 403, 455, 471, 554, 570, 622, 638, 683, 699, 751, 767, 776, 792, 844, 860, 905, 921, 973, 989,
2, 18, 70, 86, 131, 147, 199, 215, 288, 304, 356, 372, 417, 433, 485, 501, 520, 536, 588, 604, 649, 665, 717, 733, 810, 826, 878, 894, 939, 955, 1007, 1023,
14, 30, 74, 90, 143, 159, 203, 219, 300, 316, 360, 376, 429, 445, 489, 505, 516, 532, 576, 592, 645, 661, 705, 721, 806, 822, 866, 882, 935, 951, 995, 1011,
44, 60, 104, 120, 173, 189, 233, 249, 270, 286, 330, 346, 399, 415, 459, 475, 550, 566, 610, 626, 679, 695, 739, 755, 772, 788, 832, 848, 901, 917, 961, 977,
36, 52, 96, 112, 165, 181, 225, 241, 262, 278, 322, 338, 391, 407, 451, 467, 558, 574, 618, 634, 687, 703, 747, 763, 780, 796, 840, 856, 909, 925, 969, 985,
6, 22, 66, 82, 135, 151, 195, 211, 292, 308, 352, 368, 421, 437, 481, 497, 524, 540, 584, 600, 653, 669, 713, 729, 814, 830, 874, 890, 943, 959, 1003, 1019,
11, 27, 79, 95, 138, 154, 206, 222, 297, 313, 365, 381, 424, 440, 492, 508, 513, 529, 581, 597, 640, 656, 708, 724, 803, 819, 871, 887, 930, 946, 998, 1014,
41, 57, 109, 125, 168, 184, 236, 252, 267, 283, 335, 351, 394, 410, 462, 478, 547, 563, 615, 631, 674, 690, 742, 758, 769, 785, 837, 853, 896, 912, 964, 980,
33, 49, 101, 117, 160, 176, 228, 244, 259, 275, 327, 343, 386, 402, 454, 470, 555, 571, 623, 639, 682, 698, 750, 766, 777, 793, 845, 861, 904, 920, 972, 988,
3, 19, 71, 87, 130, 146, 198, 214, 289, 305, 357, 373, 416, 432, 484, 500, 521, 537, 589, 605, 648, 664, 716, 732, 811, 827, 879, 895, 938, 954, 1006, 1022,
15, 31, 75, 91, 142, 158, 202, 218, 301, 317, 361, 377, 428, 444, 488, 504, 517, 533, 577, 593, 644, 660, 704, 720, 807, 823, 867, 883, 934, 950, 994, 1010,
45, 61, 105, 121, 172, 188, 232, 248, 271, 287, 331, 347, 398, 414, 458, 474, 551, 567, 611, 627, 678, 694, 738, 754, 773, 789, 833, 849, 900, 916, 960, 976,
37, 53, 97, 113, 164, 180, 224, 240, 263, 279, 323, 339, 390, 406, 450, 466, 559, 575, 619, 635, 686, 702, 746, 762, 781, 797, 841, 857, 908, 924, 968, 984,
7, 23, 67, 83, 134, 150, 194, 210, 293, 309, 353, 369, 420, 436, 480, 496, 525, 541, 585, 601, 652, 668, 712, 728, 815, 831, 875, 891, 942, 958, 1002, 1018
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


