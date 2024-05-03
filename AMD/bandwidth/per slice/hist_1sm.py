import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata 
from scipy.stats import norm
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter


# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Initialize lists to hold x, y, z values
x, y, z = [], [], []

# Read the file and extract the data
with open('result1SM.txt', 'r') as infile:
    slice_num = 0
    sm_num = 0
    for line in infile:
        line = line.strip()
        if line == '':
            slice_num += 1
            sm_num = 0
        else:
            bandwidth = float(line)
            x.append(sm_num)
            y.append(slice_num)
            z.append(bandwidth)
            sm_num += 1

# Convert lists to numpy arrays for plotting
z = np.array(z)

# Create a new figure with the specified width
fig = plt.figure(figsize=(6/2.54, 6/2.54))

min_val = 20
max_val = 45

# Calculate median and standard deviation
median = np.median(z)
std_dev = np.std(z)

# Calculate the number of bins
num_bins = int((max_val - min_val) / 0.5)  # Bin width of 0.5

# Calculate the bin edges
bin_edges = np.linspace(min_val, max_val, num_bins+1)

# Get the current axes
ax = plt.gca()

# Create the histogram
counts, bins = np.histogram(z, bins=bin_edges)

# Convert counts to percentages
counts = 100 * counts / counts.sum()

# Plot the bars
plt.bar(bins[:-1], counts, width=np.diff(bins), color='#464646', edgecolor='black', linewidth=1)

# Create a function to format the y-axis ticks
def to_percent(y, position):
    return str(y) + '%'

# Create a formatter
formatter = FuncFormatter(to_percent)

# Set the y-axis formatter
plt.gca().yaxis.set_major_formatter(formatter)

# Set the y-label to 'Frequency'
plt.ylabel('Frequency')

# Set the x-axis limits
ax.set_xlim(left=20, right=45)

# Set the x-ticks to be every 1 from 70 to 100
plt.xticks(np.arange(20, 45, 1))

# Set the y-axis limits
plt.ylim(0, 40)

# Set the y-ticks to be every 5 from 0 to 30
plt.yticks(np.arange(0, 41, 10))

""" # Fit a normal distribution to the data
mu, std = norm.fit(z)

# Plot the PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std) * len(z) * np.diff(bins)[0]  # Scale the PDF by the number of data points and the bin width
plt.plot(x, p, 'k', linewidth=1, color='#838383', alpha=0.6)

# Plot the median as a vertical dashed line
peak_height = max(p)
plt.vlines(mu, 0, peak_height, colors='#838383', linestyles='dashed', linewidth=0.5) """

# Get the current axes
ax = plt.gca()

# Get the x-tick labels
labels = [item.get_text() for item in ax.get_xticklabels()]

# Set the x-tick labels to be empty except for one every 5 and all between 82 and 85
for i in range(len(labels)):
    if (i % 5 != 0): #and (i < 22 or i > 25):
        labels[i] = ''

ax.set_xticklabels(labels)


# Add labels
#plt.xlabel('Bandwidth [GB/s]')
#plt.ylabel('Frequency')

plt.savefig("1sm1slicehist.png", dpi=600, bbox_inches='tight')
plt.show()