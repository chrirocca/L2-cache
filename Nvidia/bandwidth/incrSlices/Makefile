all:
	nvcc -Xptxas -dlcm=cg -O3 -gencode arch=compute_70,code=sm_70 -lcupti -lcuda -lnvidia-ml -lnvperf_host -lnvperf_target -o Leakage diffMPV100.cu