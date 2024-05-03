make

for ((j=0; j<1; j++)){
    for ((i = 0; i < 1024; i ++)) 
    do 
    ./Leakage 100 $i $j
    done
}
