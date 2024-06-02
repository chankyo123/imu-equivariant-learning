import numpy as np
from scipy.spatial.transform import Rotation
import os
import shutil
import json

world_2_body_transformation = True

# Define the paths to the source and destination directories
# source_directory = './local_data/tlio_golden'
# destination_directory = './local_data_bodyframe/tlio_golden'
# source_directory = './local_data_bodyframe_test_so3/tlio_golden'
# destination_directory = './local_data_test_so3/tlio_golden'
source_directory = './local_data_bodyframe_test_so3_2/tlio_golden'
destination_directory = './local_data_test_so3/tlio_golden'

# source_directory = './local_data_bodyframe_test_so2_fixed/tlio_golden'
# destination_directory = './local_data_test_so2_fixed/tlio_golden'
# source_directory = './local_data_bodyframe_test_so2_fixed_notcsv/tlio_golden'
# destination_directory = './local_data_test_so2_fixed_notcsv/tlio_golden'

source_directory= './sim_imu_longerseq_worldframe'
destination_directory = './sim_imu_longerseq'
# source_directory = './sim_imu_longerseq_idso3_fixed2'
# destination_directory = './sim_imu_longerseq_idso3_fixed2_2_worldframe'

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Iterate over each subdirectory in the source directory
for subdir in os.listdir(source_directory):
    subdir_path = os.path.join(source_directory, subdir+"/")

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # List all files in the subdirectory
        files = os.listdir(subdir_path)

        # Filter out files with the name 'imu0_resampled.npy'
        files = [file for file in files if file.startswith('imu0_resampled.npy')]

        # Iterate over each file
        for file in files:
            # Load the npy file
            data = np.load(os.path.join(subdir_path, file))
            # Extract quaternion data
            quaternion = data[:, 7:11]

            # Initialize lists to store rotated gyroscope and accelerometer data, and gt_velocity
            rotated_gyr_data_list = []
            rotated_acc_data_list = []
            rotated_vel_data_list = []

            # Convert quaternions to rotation matrices for all timesteps
            rotations = Rotation.from_quat(quaternion)
            rotation_matrices = rotations.as_matrix()

            # Calculate the inverse of all rotation matrices
            inverse_rotation_matrices = np.transpose(rotation_matrices, axes=(0, 2, 1))

            # Extract gyroscope, accelerometer, and velocity data for all timesteps
            gyr_data = data[:, 1:4]
            acc_data = data[:, 4:7]
            vel_data = data[:, -3:]

            # Apply inverse rotation to gyroscope, accelerometer, and velocity data
            #1. world -> body transformation
            if world_2_body_transformation:
                rotated_gyr_data = np.einsum('ijk,ik->ij', inverse_rotation_matrices, gyr_data)
                rotated_acc_data = np.einsum('ijk,ik->ij', inverse_rotation_matrices, acc_data)
                rotated_vel_data = np.einsum('ijk,ik->ij', inverse_rotation_matrices, vel_data)
            
            #2. body -> world transformation
            else:
                rotated_gyr_data = np.einsum('ijk,ik->ij', rotation_matrices, gyr_data)
                rotated_acc_data = np.einsum('ijk,ik->ij', rotation_matrices, acc_data)
                rotated_vel_data = np.einsum('ijk,ik->ij', rotation_matrices, vel_data)
            
            # Optionally, if you need lists, you can convert the arrays to lists
            rotated_gyr_data_list = rotated_gyr_data.tolist()
            rotated_acc_data_list = rotated_acc_data.tolist()
            rotated_vel_data_list = rotated_vel_data.tolist()

            # Convert lists to numpy arrays
            rotated_gyr_data_array = np.array(rotated_gyr_data_list)
            rotated_acc_data_array = np.array(rotated_acc_data_list)
            rotated_vel_data_array = np.array(rotated_vel_data_list)

            # Save the transformed data in the destination directory
            # print(data[:, :1].shape, rotated_gyr_data_array.shape, rotated_acc_data_array.shape, quaternion.shape, data[:, 11:].shape)
            os.makedirs(os.path.join(destination_directory, subdir), exist_ok=True)
            np.save(os.path.join(destination_directory, subdir, file), np.concatenate([data[:, :1], rotated_gyr_data_array, rotated_acc_data_array, quaternion, data[:, 11:-3], rotated_vel_data_array], axis=1))
            print(os.path.join(destination_directory, subdir, file))
            # print(np.concatenate([data[:, :1], rotated_gyr_data_array, rotated_acc_data_array, quaternion, data[:, 11:-3], rotated_vel_data_array], axis=1).shape)

# List of specific files to copy
specific_files = ['all_ids.txt', 'spline_metrics.csv', 'test_list.txt', 'train_list.txt', 'val_list.txt', 'test_list1.txt', 'test_list2.txt', 'test_list3.txt', 'test_list4.txt']

# Copy the specific files from the source directory to the destination directory
for file in specific_files:
    file_path = os.path.join('./local_data_bodyframe_test_so2/tlio_golden', file)
    if os.path.exists(file_path):
        shutil.copy(file_path, destination_directory)
        print(file_path)
        
        
# Copy all other files from the source directory to the destination directory
for subdir in os.listdir(source_directory):
    subdir_path = os.path.join(source_directory, subdir)
    destination_subdir = os.path.join(destination_directory, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # Iterate through files in the subdirectory
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)

            # Check if it's not an npy file
            if not file.endswith(".npy"):
                # Copy the file to the destination subdirectory
                shutil.copy(file_path, destination_subdir)
                print(file_path)
                
                # Check if it's imu0_resampled_description.json
                if file == 'imu0_resampled_description.json':
                    # Read the JSON file
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Modify the JSON data
                    if 'columns_name(width)' in json_data:
                        columns = json_data['columns_name(width)']
                        if world_2_body_transformation: 
                            if 'vel_World(3)' in columns:
                                idx = columns.index('vel_World(3)')
                                columns[idx] = 'vel_Body(3)'
                                json_data['columns_name(width)'] = columns
                        else:
                            if 'vel_Body(3)' in columns:
                                idx = columns.index('vel_Body(3)')
                                columns[idx] = 'vel_World(3)'
                                json_data['columns_name(width)'] = columns
                    
                    # Define the destination file path
                    destination_file_path = os.path.join(destination_subdir, file)
                    
                    # Write the modified JSON data to the destination subdirectory
                    with open(destination_file_path, 'w') as f:
                        json.dump(json_data, f, indent=4)  # Set the indent parameter to 4 for 4 spaces indentation






# import os

# def count_files_in_subdirectories(directory):
#     total_files = 0

#     # Iterate through subdirectories
#     for subdir, _, files in os.walk(directory):
#         total_files += len(files)

#     return total_files

# # Example usage
# directory_path = "./local_data_bodyframe/tlio_golden"

# total_files_count = count_files_in_subdirectories(directory_path)
# print(f"Total number of files in subdirectories: {total_files_count}")







# import numpy as np
# import os

# def compare_npy_files(file1, file2):
#     """Compare whether two .npy files have the same content."""
#     print(file1, file2)
#     # Load data from .npy files
#     data1 = np.load(file1)
#     data2 = np.load(file2)
    
#     diff = np.abs(data1 - data2)
#     print(data1[1,:])
#     print(data2[1,:])
#     threshold = 1e-6 
#     # Check if all differences are within the threshold
#     if np.all(diff <= threshold):
#         print(True)
#     else:
#         print(False)
#         return 


# # Directory paths
# directory1 = 'local_data/tlio_golden/137757803662107/imu0_resampled.npy'
# # directory1 = 'local_data_bodyframe/tlio_golden/137757803662107/imu0_resampled.npy'
# directory2 = 'local_data_bodyframe/tlio_golden/137757803662107/imu0_resampled.npy'

# compare_npy_files(directory1, directory2)
