import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num', type=int, required=True,
                    help='the number of the latency file to process')

args = parser.parse_args()

# Construct the filename
filename = f'latency_{args.num}.log'

# Initialize lists to hold x, y, z values
x, y, z = [], [], []

# Read the file and extract the data
with open(filename, 'r') as infile:
    for line in infile:
        line = line.strip()
        if line != '':
            bandwidth = float(line)
            z.append(bandwidth)

# Convert lists to numpy arrays for plotting
z = np.array(z)

# Create a new figure with the specified width
fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))

# Set the y-label to 'Frequency'
plt.ylabel('Latency (clock cycles)')
plt.xlabel('L2 slice')

# Set the y-axis limits
plt.ylim(bottom=250, top=320)

# Set the x-ticks to be every 4 from 250 to 350
ax.xaxis.set_major_locator(MultipleLocator(4))

# Set the x-subticks to be every 1 from 250 to 350
ax.xaxis.set_minor_locator(AutoMinorLocator(4))

# Set the y-axis limits
plt.xlim(-1, 32)

# Set the y-ticks to be every 15 from 0 to 100
ax.yaxis.set_major_locator(MultipleLocator(15))

# Set the y-subticks to be every 5 from 0 to 100
ax.yaxis.set_minor_locator(AutoMinorLocator(3))

# Plot the bar plot
plt.bar(range(len(z)), z, color='#464646')

# Save the plot
plt.savefig(f"{args.num}_bar.png", dpi=600, bbox_inches='tight')
plt.show()