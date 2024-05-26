all:
	nvcc -Xptxas -dlcm=cg -O3 -lcupti -lcuda -lnvidia-ml -lnvperf_host -lnvperf_target -o Leakage diffMP.cu