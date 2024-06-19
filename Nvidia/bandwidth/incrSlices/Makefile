all:
	nvcc -Xptxas -dlcm=cg -O3 -gencode arch=compute_80,code=sm_80 -lcupti -lcuda -lnvidia-ml -lnvperf_host -lnvperf_target -o Leakage diffMPA100.cu