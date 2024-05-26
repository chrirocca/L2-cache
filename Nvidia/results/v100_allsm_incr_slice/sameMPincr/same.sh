make

    for ((g = 0 ; g < 4; g ++)){
        for ((k = 8; k < 9; k ++)){
            for ((d = 1; d < 81; d ++))
            do 
                nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput --log-file  cluster"$g"_slices"$k"_sm"$d".log ./Leakage $g $k $d
            done
}
}


