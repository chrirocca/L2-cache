import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

colors = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal']
markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', 'D', 'd', 'X', '8']

for i in range(14,16):
    df = pd.read_csv(f'slicesA100_test{i}.log', sep='\s+', header=None)
    data = df[0]
    plt.scatter(range(1, len(data) + 1), data, color=colors[i-1], marker=markers[i-1], label=f'test{i}', s=6)

plt.xlabel('L2 Slices')
plt.ylabel('Bandwidth (GB/s)')
plt.xticks(range(0, 81, 4))
plt.yticks(range(0, 2201, 200))

legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fancybox=False, shadow=False, ncol=4, fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')

plt.savefig('bandwidthAll.png', dpi = 600, bbox_inches='tight')
plt.show()