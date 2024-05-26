import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Initialize a dictionary to hold z values for each SM
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
            #Ignore slices from 77 to 79
            if 77 <= slice_num <= 79:
                continue
            #if 38 <= slice_num <= 79:
            #    continue
            bandwidth = float(line)
            if sm_num not in z_dict:
                z_dict[sm_num] = []
            z_dict[sm_num].append((slice_num, bandwidth))
            sm_num += 1

# Create a separate plot for each SM
for sm_num, data in z_dict.items():
    # Convert list to numpy array for plotting
    data = np.array(data)

    # Create a new figure with the specified width
    fig = plt.figure(figsize=(8.5/2.54, 4/2.54))

    # Get the current axes
    ax = plt.gca()

    # Plot the data as a bar plot
    plt.bar(data[:, 0], data[:, 1], color='#464646')

    # Set the x-label to 'Slice Number'
    plt.xlabel('L2 slice')

    # Set the y-label to 'Bandwidth'
    plt.ylabel('Bandwidth (GB/s)')

    # Set the x-ticks to be every 8
    plt.xticks(np.arange(0, np.max(data[:, 0])+1, 8))

    # Add subticks every 1 for the x-axis from 0 to 80
    ax.set_xticks(np.arange(0, 77, 1), minor=True)

    # Set the y-ticks to be every 5 and y-axis from 0 to 45
    plt.yticks(np.arange(0, 51, 10))

    # Calculate average and standard deviation
    avg = np.mean(data[:, 1])
    std_dev = np.std(data[:, 1])

 # Add average and standard deviation to the figure
    #plt.text(0.02, 0.95, f'Average: {avg:.2f} GB/s\nStd Dev: {std_dev:.2f} GB/s', transform=ax.transAxes, verticalalignment='top')

    # Add title
    #plt.title(f'SM {sm_num}', fontsize=14, fontname='DejaVu Sans')

    plt.savefig(f"sm_{sm_num}.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Close the figure
    plt.close(fig)