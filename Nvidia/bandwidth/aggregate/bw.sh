make

for ((i = 32; i < 33; i ++)){
    for ((l=8; l < 9; l ++))
do
    ./Leakage $i $l 10
done
}