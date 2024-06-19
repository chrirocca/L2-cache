import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Read the files, each line is a different data point
df_A100 = pd.read_csv('slicesA100_1GPC.log', sep='\s+', header=None)
df_V100 = pd.read_csv('slicesV100_1GPC.log', sep='\s+', header=None)


# Extract the first column
data_A100 = df_A100[0]
data_V100 = df_V100[0]

# Plot the data
plt.scatter(range(1, len(data_A100) + 1), data_A100, color='black', marker='o', label='A100', s=6)
plt.scatter(range(1, len(data_V100) + 1), data_V100, color='gray', marker='x', label='V100', s=6)

# Fit a 2nd degree polynomial to the data and plot the trend line for A100
z = np.polyfit(range(1, len(data_A100) + 1), data_A100, 2)
p = np.poly1d(z)
#plt.plot(range(1, len(data_A100) + 1), p(range(1, len(data_A100) + 1)), ':', color='black')

# Fit a logistic function to the data and plot the trend line for V100
popt, pcov = curve_fit(logistic, range(1, len(data_V100) + 1), data_V100, p0=[np.max(data_V100), 1, np.median(range(1, len(data_V100) + 1))])
#plt.plot(range(1, len(data_V100) + 1), logistic(range(1, len(data_V100) + 1), *popt), 'k--', color='gray')

# rest of the code remains the same

# Add labels and title
plt.xlabel('L2 Slices')
plt.ylabel('Bandwidth (GB/s)')

# Set x-axis and y-axis limits and ticks
plt.xticks(range(0, 81, 4))  # Show one tick every 4 for x-axis
plt.yticks(range(0, 601, 100))  # Show from 0 to 2400 for y-axis, one tick every 400

# Add legend outside the plot with a black border
legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fancybox=False, shadow=False, ncol=4, fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')

# Save the plot as a PNG file
plt.savefig('bandwidthGPC.png', dpi = 600, bbox_inches='tight')

# Show the plot
plt.show()