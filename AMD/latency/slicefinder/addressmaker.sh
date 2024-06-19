make

for ((j=0; j<1; j++)){
    for ((i = 256; i < 288; i ++)) 
    do 
    omniperf profile -n 'latency_'$j'_'$i -- ./hip-latency $i $j
    omniperf analyze -n per_second -p workloads/'latency_'$j'_'$i/MI100/ > 'latency_'$j'_'$i.log
    done
}

#python3 extract.py --sm_range 0 1 --address_range 0 32