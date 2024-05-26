make

    for ((g = 0 ; g < 4; g ++)){
        for ((k = 1; k < 9; k ++))
            do 
                nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput --log-file  cluster"$g"_slices"$k".log ./Leakage $g $k 
            done
}



