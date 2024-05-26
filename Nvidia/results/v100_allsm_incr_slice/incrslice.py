import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Initialize lists to store the data
diffMP_data = []
sameMP_data = []

# Read the data from the diffMP files
for i in range(1, 9):
    df = pd.read_csv(f'diffMP/slices{i}.log', delim_whitespace=True, header=None, skiprows=6, on_bad_lines='skip')
    if df.shape[0] > 0:
        diffMP_data.append(df.iloc[0, -2])

# Read the data from the sameMP files
for i in range(1, 9):
    df = pd.read_csv(f'sameMP/cluster0_slices{i}.log', delim_whitespace=True, skiprows=10, header=None, on_bad_lines='skip')
    if df.shape[0] > 0:
        value = df.iloc[0, -1]
        if isinstance(value, str):
            numeric_part = re.findall('(\d+\.?\d*)', value)[0]
            sameMP_data.append(pd.to_numeric(numeric_part))

plt.figure(figsize=(8/2.54, 6/2.54))

# Create the bar plot
x = np.arange(1, 9)
width = 0.35  # the width of the bars

plt.bar(x - width/2, sameMP_data, width, color='#404040', edgecolor='black', linewidth=1, label='Contiguous MP')
plt.bar(x + width/2, diffMP_data, width, color='#808080', edgecolor='black', linewidth=1, label='Distributed MP')


# Set the x and y axis labels
plt.xlabel('Slices accessed')
plt.ylabel('Bandwidth (GB/s)')

# Set the y axis limits and ticks
plt.ylim(0, 800)
plt.yticks(range(0, 801, 200))
plt.xticks(range(0, 9, 1))

# Add a legend
legend = plt.legend(loc='lower center', bbox_to_anchor=(0.45, 1.0), fancybox=False, shadow=False, ncol=1, fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')

# Save the plot as a PNG file
plt.savefig('slices.png', dpi = 600, bbox_inches='tight')
