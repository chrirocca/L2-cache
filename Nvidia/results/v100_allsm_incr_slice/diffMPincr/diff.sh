make

        for ((k = 8; k < 9; k ++)){
        for ((j = 1; j < 81; j ++))
            do 
                nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput --log-file  slices"$k"_sms"$j".log ./Leakage $k $j
            done

        }


