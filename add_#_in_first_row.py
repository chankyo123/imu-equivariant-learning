import csv
import os
import glob

# Define the directory path and search pattern
directory_path = 'sim_imu_longerseq'
search_pattern = os.path.join(directory_path, '*/imu_samples_0.csv')

# Find all files matching the pattern
input_files = glob.glob(search_pattern)
for input_file_path in input_files:
    # Define the output file path
    output_file_path = os.path.join(os.path.dirname(input_file_path), 'imu_samples_calibrated.csv')

    # Read the CSV file
    with open(input_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Add '#' to the first row and format the header
    if rows:
        rows[0] = ['#' + rows[0][0]] + [f" {col}" for col in rows[0][1:]]

    # Format the data rows and add space after comma
    for i in range(1, len(rows)):
        # Convert the first two columns to integers
        rows[i][0] = int(float(rows[i][0]))  # Convert to float first to handle scientific notation
        rows[i][1] = int(float(rows[i][1]))

        # Format the remaining columns to seven digits after the decimal point
        for j in range(2, len(rows[i])):
            rows[i][j] = f" {float(rows[i][j]):.7f}"

    # Write the modified data to a new CSV file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)

    print(f"Modified CSV has been saved to {output_file_path}")
