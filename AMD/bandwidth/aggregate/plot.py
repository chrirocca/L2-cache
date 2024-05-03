import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Read the files, skipping the first 3 lines
df1 = pd.read_csv('1warps.txt', delim_whitespace=True, header=None, skiprows=3)
df8 = pd.read_csv('8warps.txt', delim_whitespace=True, header=None, skiprows=3)
df1_global = pd.read_csv('global memory/1warp.txt', delim_whitespace=True, header=None)
df8_global = pd.read_csv('global memory/8warp.txt', delim_whitespace=True, header=None)

# Extract the first column
data1 = df1[0]
data8 = df8[0]
data1_global = df1_global[0]
data8_global = df8_global[0]

# Set the figure size to be 8cm x 16cm (converted to inches)
plt.figure(figsize=(16/2.54, 8/2.54))

# Plot the data
plt.scatter(range(1, len(data1) + 1), data1, color='black', marker='o', label='1 WARP')
plt.scatter(range(1, len(data8) + 1), data8, color='gray', marker='x', label='8 WARP')
plt.scatter(range(1, len(data1_global) + 1), data1_global, color='black', marker='o')
plt.scatter(range(1, len(data8_global) + 1), data8_global, color='gray', marker='x')

# Fit a 2nd degree polynomial to the data and plot the trend line for 1 WARP
z = np.polyfit(range(1, len(data1) + 1), data1, 2)
p = np.poly1d(z)
plt.plot(range(1, len(data1) + 1), p(range(1, len(data1) + 1)), 'k:', color='black', label='L2 Cache')

# Define the logistic function
def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

popt, pcov = curve_fit(logistic, range(1, len(data1_global) + 1), data1_global, p0=[np.max(data1_global), 1, np.median(range(1, len(data1_global) + 1))])
plt.plot(range(1, len(data1_global) + 1), logistic(range(1, len(data1_global) + 1), *popt), 'k--', color='black', label='Global Memory')

# Fit a logistic function to the data and plot the trend line for 8 WARP
popt, pcov = curve_fit(logistic, range(1, len(data8) + 1), data8, p0=[np.max(data8), 1, np.median(range(1, len(data8) + 1))])
plt.plot(range(1, len(data8) + 1), logistic(range(1, len(data8) + 1), *popt), 'k:', color='gray')

# Fit a logistic function to the data and plot the trend line for 8 WARP (Global Memory)
popt, pcov = curve_fit(logistic, range(1, len(data8_global) + 1), data8_global, p0=[np.max(data8_global), 1, np.median(range(1, len(data8_global) + 1))], maxfev=150000)
plt.plot(range(1, len(data8_global) + 1), logistic(range(1, len(data8_global) + 1), *popt), 'k--', color='gray')

# Add labels and title
plt.xlabel('CTAs per SM')
plt.ylabel('Bandwidth (GB/s)')

# Set x-axis and y-axis limits and ticks
plt.xticks(range(0, 33, 4))  # Show one tick every 4 for x-axis
plt.yticks(range(0, 2401, 400))  # Show from 0 to 2400 for y-axis, one tick every 400

# Add legend outside the plot with a black border
legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fancybox=False, shadow=False, ncol=4, fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')

# Save the plot as a PNG file
plt.savefig('bandwidth.png', dpi = 600, bbox_inches='tight')

# Show the plot
plt.show()