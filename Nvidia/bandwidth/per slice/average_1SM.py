import re

with open('result.txt', 'w') as outfile:
    for slice_num in range(32):
        for sm_num in range(80):
            numbers = []  # List to store the numbers for each iteration
            for iteration in range(1, 10):
                filename = f"slice{slice_num}_sm{sm_num}iteration{iteration}.log"
                try:
                    with open(filename, 'r') as infile:
                        lines = infile.readlines()
                        if len(lines) > 10:
                            line = lines[10]
                            last_word = line.split()[-1]
                            number = re.sub(r'[^\d.]', '', last_word)
                            numbers.append(float(number))  # Add the number to the list
                except FileNotFoundError:
                    print(f"File {filename} not found.")
            # Calculate the average and write it to the output file
            if numbers:  # Check if the list is not empty
                average = sum(numbers) / len(numbers)
                if average > 34:
                    print(f"Slice {slice_num}, SM {sm_num}: average {average} is over 34")
                outfile.write(str(average) + '\n')
        outfile.write('\n')