import matplotlib.pyplot as plt
import numpy as np

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
            # Ignore slices from 77 to 79
            if 77 <= slice_num <= 79:
                continue
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
    fig = plt.figure(figsize=(12/2.54, 6/2.54))

    # Get the current axes
    ax = plt.gca()

    # Plot the data as a bar plot
    plt.bar(data[:, 0], data[:, 1], color='#464646')

    # Set the x-label to 'Slice Number'
    plt.xlabel('L2 slice')

    # Set the y-label to 'Bandwidth'
    plt.ylabel('Bandwidth (GB/s)')

    # Set the x-ticks to be every 8
    plt.xticks(np.arange(0, np.max(data[:, 0])+1, 5))

    # Set the y-ticks to be every 5 and y-axis from 0 to 45
    plt.yticks(np.arange(0, 51, 5))

    plt.savefig(f"sm_{sm_num}.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Close the figure
    plt.close(fig)