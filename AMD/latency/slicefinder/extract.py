import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--sm_range', type=int, nargs=2, required=True,
                    help='the range of SMs (inclusive start, exclusive end)')
parser.add_argument('--address_range', type=int, nargs=2, required=True,
                    help='the range of addresses (inclusive start, exclusive end)')

args = parser.parse_args()

# Define the range of SMs and addresses
sm_range = range(*args.sm_range)
address_range = range(*args.address_range)

# Initialize a 2D list to hold the values for all SMs
all_values = []

# Iterate over each SM
for sm in sm_range:
    # Initialize a list to hold the values for this SM
    sm_values = []
    # Iterate over each address
    for address in address_range:
        # Construct the filename
        filename = f'latency_{sm}_{address}.log'
        # Check if the file exists
        if os.path.exists(filename):
            # Open the file
            with open(filename, 'r') as infile:
                # Skip to line 1172
                for _ in range(1171):
                    next(infile)
                # Read lines 1172 to 1234
                for i, line in enumerate(infile, start=1172):
                    if i > 1234:
                        break
                    # Only process even lines
                    if i % 2 == 0:
                        # Split the line into numbers
                        numbers = line.split('│')
                        # Check the 4th number
                        if float(numbers[4]) > 50000.0:
                            # Add the 2nd number to the list as an integer
                            sm_values.append(int(float(numbers[2].strip())))
    # Add the values for this SM to the 2D list
    all_values.append(sm_values)

# Open the output file
with open('address.log', 'w') as outfile:
    # Transpose the 2D list and write the values to the output file
    for values in zip(*all_values):
        outfile.write('\t'.join(map(str, values)) + '\n')