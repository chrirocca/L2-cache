make
for ((j = 0; j < 128; j ++))
        do
            omniperf profile -n 'allSM-'$j -- ./hip-latency $j
            omniperf analyze -n per_second -p workloads/'allSM-'$j/MI100 > 'allSM-'$j.log
        done

