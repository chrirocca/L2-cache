make
for ((i = 0; i < 1; i ++)){

    for ((k = 32; k < 33; k ++))
        do 
            nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput,l2_read_transactions --log-file  $i'_warp_max'$k.log ./Leakage $i $k
        done
}

