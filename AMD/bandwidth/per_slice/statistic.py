import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import re

# Initialize an empty list to store the data
data = []

# Specify the log folder
log_folder = "logs"

# Loop over the range of l2_slice values
for l2_slice in range(32):
    log_file_name = f"{l2_slice}_warp_max1.log"
    data_path = os.path.join(log_folder, log_file_name)

    # Read the last value from line 11 of the log file
    with open(data_path, "r") as file:
        lines = file.readlines()
        last_line = lines[10].strip()  # Line 11 is index 10 in the list
        last_value = last_line.split()[-1]  # Extract the last value from the line
        last_number = float(re.findall(r'\d+\.\d+', last_value)[-1])  # Extract the last number from the line

        # Append the last number to the data list
        data.append(last_number)

# Calculate the median, variance, and standard deviation
median = np.median(data)
variance = np.var(data)
std_dev = np.sqrt(variance)

# Print the results
print(f"Median: {median} GB/s")
print(f"Variance: {variance} (GB/s)^2")
print(f"Standard Deviation: {std_dev} GB/s")
