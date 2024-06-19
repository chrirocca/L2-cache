make

rm *.log

for ((j=0; j<1; j++)){
    for ((i = 0; i < 32; i ++)) 
    do 
    ./hip-latency $i $j >> 'latency_'$j.log
    #python3 preplot.py --num $j
    #python3 plot.py --num $j
    done
}
