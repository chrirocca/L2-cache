make

for ((i = 32; i < 33; i ++)){
    for ((l=1; l < 2; l ++))
do
    ./Leakage $i $l 10
done
}