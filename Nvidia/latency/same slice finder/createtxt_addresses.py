import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import glob, os
import re
import sys


from csv import reader

f= open("addresses.txt","w+")
plot_idx = 0
x = np.arange(32)

partition = {}
access_count = {}

grandezza = int(sys.argv[1])

for j in range (grandezza):

    select_num = 0
    file_name = str(j) +".log"
    file = open(file_name, 'r')
    print ("sto leggendo %s" % (file_name))
    lines = file.readlines()
    plot_idx = 0

    for line in islice(lines, 20, 22):
        select_num = 555
        num_with_space = line.split('[', 1)[1].split(']')[0]
        num_list = re.findall(r'\S+', num_with_space)  
    
        i = 0
        partition[plot_idx] = []
        access_count[plot_idx] = []
        for num in num_list:
            partition[plot_idx].append(i)
            int_num = int(num)
            access_count[plot_idx].append(int_num)
            if int(num) > 0:
                select_num = i
            i += 1

        if (select_num != 555):
            f.write("L2 block %d, partition %d, start_idx %d\n" % (select_num, plot_idx, j ))
        
        plot_idx += 1



    
    file.close()


f.close()

        

