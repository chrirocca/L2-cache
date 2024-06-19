make
for ((i = 0; i < 1; i ++)){

for ((j = 0; j < 80; j ++)){

    for ((k = 32; k < 33; k ++))
        do
        for ((l = 0; l < 1; l ++)){ 
            nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput,l2_read_transactions --log-file  'slice'$i'_sm'$j'iteration'$l.log ./Leakage $i $k $j
        }
        done
}
}

