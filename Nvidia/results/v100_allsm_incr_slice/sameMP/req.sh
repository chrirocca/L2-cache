make

    for ((g = 0 ; g < 4; g ++)){
        for ((k = 1; k < 9; k ++))
            do 
                nvprof --aggregate-mode off -e all --log-file cluster"$g"_slices"$k".log ./Leakage $g $k 
            done
}



