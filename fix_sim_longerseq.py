import os
import numpy as np

def fix_acc_data_in_npy_files(root_dir):
    """
    Iterate through all .npy files in each subdirectory, apply the required modifications 
    to the accelerometer data, and save the modified data back to the same file.

    Parameters:
    root_dir (str): Path to the root directory containing subdirectories with .npy files.

    Returns:
    None
    """
    # Iterate through all subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'imu0_resampled.npy':
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {file_path}")
                
                # Load the .npy file
                data = np.load(file_path)
                
                # Apply the modifications to the accelerometer data
                data[:, 4:7] = -data[:, 4:7]
                data[:, 4:7] = data[:, 4:7] + np.array([0, 0, 2*9.81])
                
                # Save the modified data back to the same file
                np.save(file_path, data)

# Example usage
root_directory = "sim_imu_longerseq_worldframe"  # Replace with the path to your root directory
fix_acc_data_in_npy_files(root_directory)
