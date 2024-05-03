make

for ((i = 1; i < 15; i ++))
do
    nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput,l2_read_transactions --log-file result"$i".log ./Leakage 1 32 10 0 $i
done