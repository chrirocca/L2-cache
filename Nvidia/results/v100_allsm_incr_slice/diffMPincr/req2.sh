make

        for ((k = 1; k < 9; k ++))
            do 
                nvprof --aggregate-mode off -e all --log-file slices"$k".log ./Leakage $k 
            done



