make

for ((i = 1; i < 15; i ++))
do
    ./Leakage 1 32 10 0 $i
done