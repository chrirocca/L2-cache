all:
	nvcc -arch=sm_70 -Xptxas -dlcm=cg -O3 -lcupti -lcuda -lnvidia-ml -lnvperf_host -lnvperf_target -o Leakage bandwidth_read.cu