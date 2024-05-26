def copy_selected_lines(cases):
    hSM_ids = {
        0: [0, 1, 16, 17, 32, 33, 48, 49, 124, 125, 126, 127, 128, 129, 130, 131],
        1: [2, 3, 18, 19, 34, 35, 50, 51, 64, 65, 78, 79, 92, 93, 106, 107],
        2: [4, 5, 20, 21, 36, 37, 52, 53, 66, 67, 80, 81, 94, 95, 108, 109],
        3: [6, 7, 22, 23, 37, 39, 54, 55, 68, 69, 82, 83, 96, 97, 110, 111],
        4: [8, 9, 24, 25, 40, 41, 56, 57, 70, 71, 84, 85, 98, 99, 112, 113],
        5: [10, 11, 26, 27, 42, 43, 58, 59, 72, 73, 86, 87, 100, 101, 114, 115],
        6: [12, 13, 28, 29, 44, 45, 60, 61, 74, 75, 88, 89, 102, 103, 116, 117, 120, 121],
        7: [14, 15, 30, 31, 46, 47, 62, 63, 76, 77, 90, 91, 104, 105, 118, 118, 122, 123],
    }

    with open('result1SM.txt', 'r') as infile, open('part0.txt', 'w') as outfile:
        for i, line in enumerate(infile):
            for case in cases:
                if i % 133 in hSM_ids[case]:
                    outfile.write(line)
                    break  # Avoid writing the same line multiple times
            if i % 133 == 132:  # After 108 lines, reset the line counter
                outfile.write('\n')  # Add a new line
                i = 0

# Chiamiamo la funzione con il caso desiderato
#copy_selected_lines([0,1,5,7])
copy_selected_lines([2,3,4,6])