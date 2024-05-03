make 

for ((i = 0; i < 6144; i ++)) 
do 
    nvprof --aggregate-mode off -e all --log-file  $i.log ./Leakage 100 $i
done

    python3 createtxt_addresses.py 6144