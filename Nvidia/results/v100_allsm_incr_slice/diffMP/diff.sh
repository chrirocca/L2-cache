make

        for ((k = 1; k < 9; k ++))
            do 
                nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput --print-gpu-trace --log-file  slices"$k".log ./Leakage $k 
            done




