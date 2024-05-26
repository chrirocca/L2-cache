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

# Initialize a dictionary to hold z values for each slice_num
z_dict = {}

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
            if slice_num not in z_dict:
                z_dict[slice_num] = []
            z_dict[slice_num].append(bandwidth)
            sm_num += 1

# Create a separate histogram for each slice_num
for slice_num, z in z_dict.items():
    # Convert list to numpy array for plotting
    z = np.array(z)

    # Create a new figure with the specified width
    fig = plt.figure(figsize=(4.5/2.54, 4/2.54))

    min_val = 20
    max_val = 45

    # Calculate median, average and standard deviation
    median = np.median(z)
    avg = np.mean(z)
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
    plt.xticks(np.arange(20, 51, 10))

    # Add subticks every 1 for the x-axis from 0 to 80
    ax.set_xticks(np.arange(20, 51, 1), minor=True)


    # Set the y-axis limits
    plt.ylim(0, 40)

    # Set the y-ticks to be every 5 from 0 to 30
    plt.yticks(np.arange(0, 41, 10))

    # Get the current axes
    ax = plt.gca()

    # Get the x-tick labels
    labels = [item.get_text() for item in ax.get_xticklabels()]

    ax.set_xticklabels(labels)

    # Add average and standard deviation to the plot
    #plt.text(0.02, 0.95, f'Average = {avg:.2f}\nStd Dev = {std_dev:.2f}', transform=ax.transAxes, verticalalignment='top')

    plt.savefig(f"slice_{slice_num}.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Close the figure
    plt.close(fig)