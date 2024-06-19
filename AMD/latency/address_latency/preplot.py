import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num', type=int, required=True,
                    help='the number of the latency file to process')

args = parser.parse_args()

# Construct the filename
filename = f'latency_{args.num}.log'

# Open the file in read mode and read lines into a list
with open(filename, 'r') as file:
    lines = file.readlines()

# Open the file in write mode to overwrite it
with open(filename, 'w') as file:
    # Initialize variables
    slice_values = []
    slice_averages = []

    # Process each line
    for line in lines:
        # Remove leading/trailing whitespace and convert to integer
        value = int(line.strip())

        # If value is over 900, start a new slice
        if value > 900:
            # If there are values in the current slice, calculate the average
            if slice_values:
                slice_average = sum(slice_values) / len(slice_values)
                slice_averages.append(slice_average)
                slice_values = []
            # Write a blank line to the file
            file.write('\n')
        else:
            # Add the value to the current slice
            slice_values.append(value)

    # If there are values in the last slice, calculate the average
    if slice_values:
        slice_average = sum(slice_values) / len(slice_values)
        slice_averages.append(slice_average)

    # Overwrite the file with the slice averages
    file.seek(0)
    file.truncate()
    for slice_average in slice_averages:
        file.write(f'{slice_average}\n')

# Read the address.log file into a list of line numbers
with open('../slicefinder/address.log', 'r') as file:
    addresses = [int(line.strip()) for line in file]

# Reorder the lines in the latency_num.log file based on the address.log file
with open(filename, 'r') as file:
    lines = file.readlines()

with open(filename, 'w') as file:
    for address in addresses:
        file.write(lines[address])