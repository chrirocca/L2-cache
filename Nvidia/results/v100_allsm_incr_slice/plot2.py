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
contiguosSMincr = []
diffSMincr = []

# Read the data from the specified files
for i in range(1, 5):
    df = pd.read_csv(f'../1GPCtoIncMP/num{i}_clusters0123_slices8_GPC0_SM14.log', delim_whitespace=True, header=None, skiprows=10)
    if df.shape[0] > 0:
        value = df.iloc[0, -1]
        if isinstance(value, str):
            numeric_part = re.findall('(\d+\.?\d*)', value)[0]
            contiguosSMincr.append(pd.to_numeric(numeric_part))

for i in range(1, 5):
    df = pd.read_csv(f'../1GPCtoIncMP/distributed_SMs/num{i}_clusters0123_slices8_SM14.log', delim_whitespace=True, header=None, skiprows=10)
    if df.shape[0] > 0:
        value = df.iloc[0, -1]
        if isinstance(value, str):
            numeric_part = re.findall('(\d+\.?\d*)', value)[0]
            diffSMincr.append(pd.to_numeric(numeric_part))

# Read the data from the diffMP files
for i in range(14, 30, 14):
    df = pd.read_csv(f'diffMPincr/slices8_sms{i}.log', delim_whitespace=True, header=None, skiprows=10, on_bad_lines='skip')
    if df.shape[0] > 0:
        value = df.iloc[0, -1]
        if isinstance(value, str):
            numeric_part = re.findall('(\d+\.?\d*)', value)[0]
            diffMP_data.append(pd.to_numeric(numeric_part))

# Read the data from the sameMP files
for i in range(14, 30, 14):
    df = pd.read_csv(f'sameMPincr/cluster0_slices8_sm{i}.log', delim_whitespace=True, skiprows=10, header=None, on_bad_lines='skip')
    if df.shape[0] > 0:
        value = df.iloc[0, -1]
        if isinstance(value, str):
            numeric_part = re.findall('(\d+\.?\d*)', value)[0]
            sameMP_data.append(pd.to_numeric(numeric_part))

# Create the figure
plt.figure(figsize=(12/2.54, 6/2.54))

# Create the first subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot

# Create the first bar plot
x = np.arange(1, 3)  # Adjusted to have only two bars
width = 0.35  # the width of the bars

plt.bar(x - width/2, sameMP_data, width, color='#404040', edgecolor='black', linewidth=1, label='Contiguous MP')
plt.bar(x + width/2, diffMP_data, width, color='#808080', edgecolor='black', linewidth=1, label='Distributed MP')


# Set the y axis limits and ticks
plt.ylim(0, 800)
# Get current y ticks
yticks = plt.yticks()[0]

# Set y ticks but remove labels
plt.yticks(yticks, labels=[])

# Set the x axis ticks and labels
plt.xticks([1, 2], ['14 SMs', '28 SMs'])


# Create the second subplot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot

x = np.arange(1, 5)  # Adjusted to have only four bars

# Create the second bar plot (you can replace this with your own data)
plt.bar(x - width/2, contiguosSMincr, width, color='#404040', edgecolor='black', linewidth=1, label='Contiguous MP')
plt.bar(x + width/2, diffSMincr, width, color='#808080', edgecolor='black', linewidth=1, label='Distributed MP')

# Set the y axis limits and ticks
plt.ylim(0, 800)
# Get current y ticks
yticks = plt.yticks()[0]

# Set y ticks but remove labels
plt.yticks(yticks, labels=[])

# Set the x axis ticks and labels
plt.xticks(range(5), ['0', '1', '2', '3', '4'])
plt.xlabel('MPs accessed')

plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.8, wspace=0.3, hspace=0)

# Add a legend for the entire figure
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(-0.1, 1), fancybox=False, shadow=False, ncol=1, fontsize=12)

# Set the border color of the legend
frame = legend.get_frame()
frame.set_edgecolor('black')

# Save the plot as a PNG file
plt.savefig('plot2.png', dpi = 600, bbox_inches='tight')
#print (hello world)