make
for ((j = 0; j < 128; j ++)){
    for ((k = 0; k < 32; k ++))
        do
            omniperf profile -n '1SM-'$j'-'$k -- ./hip-latency $k $j
            omniperf analyze -n per_second -p workloads/'1SM-'$j'-'$k/MI100 > '1SM-'$j'-'$k.log
        done
}


