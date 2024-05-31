import os
import numpy as np
from scipy.spatial.transform import Rotation
import shutil
import pandas as pd

# def rand_rotation_matrix(deflection=1.0, randnums=None):
# def rand_rotation_matrix(theta):
def rand_rotation_matrix(theta, roll, pitch):
    # """
    # Creates a random rotation matrix.
    
    # deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    # rotation. Small deflection => small perturbation.
    # randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    # """
    # # if randnums is None:
    # #     randnums = np.random.uniform(size=(3,))
        
    # theta, phi, z = randnums
    
    # theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    # phi = phi * 2.0*np.pi  # For direction of pole deflection.
    # z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # # Compute a vector V used for distributing points over the sphere
    # # via the reflection I - V Transpose(V).  This formulation of V
    # # will guarantee that if x[1] and x[2] are uniformly distributed,
    # # the reflected points will be uniform on the sphere.  Note that V
    # # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    # r = np.sqrt(z)
    # Vx, Vy, Vz = V = (
    #     np.sin(phi) * r,
    #     np.cos(phi) * r,
    #     np.sqrt(2.0 - z)
    #     )
    
    # st = np.sin(theta)
    # ct = np.cos(theta)
    
    # R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    # M = (np.outer(V, V) - np.eye(3)).dot(R)
    
    
    # #so2 rotation matrix
    # # theta = np.random.uniform(0, 2 * np.pi)
    # # theta = np.pi/2*deflection
    
    # cos_theta = np.cos(theta)
    # sin_theta = np.sin(theta)
    
    # # Create the 2D rotation matrix
    # M = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0 , 0, 1]])
    
    #so3 rotation matrix
    # theta = np.random.uniform(0, 2 * np.pi)
    # theta = np.pi/2*deflection
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    
    # Create the 2D rotation matrix
    M = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0 , 0, 1]])
    Mx = np.array([[1, 0, 0],[0, c_r, -s_r],[0,s_r, c_r]])
    My = np.array([[c_p, 0, s_p],[0, 1, 0],[-s_p, 0, c_p]])
    
    # return M
    return M @ My @ Mx

# Function to apply 3D rotation to data
def apply_rotation(data, rotation_matrix):
    rotated_data = np.matmul(data, rotation_matrix)
    return rotated_data

def copy_files_from_other_list(source_directory, all_list_path, destination_directory):
    # Read the test_list.txt file
    with open(all_list_path, 'r') as file:
        all_list = file.read().splitlines()
        
    test_list = [x for x in all_list if x not in subdirectories]
    print(len(test_list))
    # Iterate through subdirectories in the source directory
    for subdir in test_list:
        subdir_path = os.path.join(source_directory, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Iterate through files in the subdirectory
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)

                # Create the destination subdirectory if it doesn't exist
                destination_subdirectory = os.path.join(destination_directory, subdir)
                os.makedirs(destination_subdirectory, exist_ok=True)

                # Copy the file to the destination subdirectory
                shutil.copy(file_path, destination_subdirectory)


# Directory containing data
data_directory = "./local_data_bodyframe/tlio_golden"
save_directory = "./local_data_bodyframe_test_so3_csv/tlio_golden"
# data_directory = "./sim_imu_longerseq_worldframe"
# save_directory = "./sim_imu_longerseq_worldframe_idso3"

# data_directory = "./sim_imu_longerseq"
# save_directory = "./sim_imu_longerseq_idso2_fixed2"

# train_list_path = os.path.join(data_directory, "train_list.txt") 
train_list_path = os.path.join(data_directory, "test_list.txt") 
# test_list_path = os.path.join(data_directory, "test_list.txt") 
all_list_path = os.path.join(data_directory, "all_ids.txt") 

# Load train_list.txt to get subdirectory names
subdirectories1 = []
subdirectories2 = []
with open(train_list_path, "r") as file:
    subdirectories1 = [line.strip() for line in file]
# with open(test_list_path, "r") as file:
#     subdirectories2 = [line.strip() for line in file]
subdirectories = subdirectories1 + subdirectories2    
# for idx, subdirectory in subdirectories:
for idx, subdirectory in enumerate(subdirectories):
    subdirectory_path = os.path.join(data_directory, subdirectory)

    npy_file_path = os.path.join(subdirectory_path, "imu0_resampled.npy")
    if os.path.exists(npy_file_path):
        original_data = np.load(npy_file_path)

        # Define 3D rotation matrix (modify as needed)
        # rotation_matrix = Rotation.from_euler('xyz', [15, 30, 45], degrees=True).as_matrix()
        random_theta = np.random.uniform(0, 2 * np.pi, len(subdirectories))
        random_roll = np.random.uniform(0, 2 * np.pi, len(subdirectories))
        random_pitch = np.random.uniform(0, 2 * np.pi, len(subdirectories))
        
        #so2 rotation : roll == pitch == 0
        # random_roll[idx] = 0
        # random_pitch[idx] = 0
        
        # rotation_matrix = rand_rotation_matrix(random_theta[idx])
        # neg_rotation_matrix = rand_rotation_matrix(-1*random_theta[idx])

        rotation_matrix = rand_rotation_matrix(random_theta[idx], random_roll[idx], random_pitch[idx])
        neg_rotation_matrix = rand_rotation_matrix(-1*random_theta[idx], random_roll[idx], random_pitch[idx])
        
        # Apply rotation to gyroscope, acceleration, quaternion, position, and velocity data
        rotated_gyroscope = apply_rotation(original_data[:, 1:4], rotation_matrix)
        rotated_acceleration = apply_rotation(original_data[:, 4:7], rotation_matrix)
        quaternion = original_data[:, 7:11]
        r = Rotation.from_quat(quaternion)
        r = r.as_matrix()
        # rotated_quaternion = apply_rotation(r, rotation_matrix)
        rotated_quaternion = apply_rotation(neg_rotation_matrix,r)
        r = Rotation.from_matrix(rotated_quaternion)
        rotated_quaternion = r.as_quat()
        # print(rotated_quaternion[0,:])
        
        rotated_position = apply_rotation(original_data[:, 11:14], rotation_matrix)
        rotated_velocity = apply_rotation(original_data[:, 14:17], rotation_matrix)

        # Concatenate the time column and rotated data
        rotated_data = np.concatenate([original_data[:, :1], rotated_gyroscope, rotated_acceleration,
                                    rotated_quaternion, rotated_position, rotated_velocity], axis=1)
        
        # Save the rotated data in the same format
        path = os.path.join(save_directory, subdirectory)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "imu0_resampled.npy"), rotated_data)
    
    csv_file_path = os.path.join(subdirectory_path, "imu_samples_0.csv")
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        if df.shape[1] >= 7:  # Ensure the CSV file has enough columns
            gyro_data = df.iloc[:, 2:5].values
            acc_data = df.iloc[:, 5:8].values
            
            # Apply rotation to gyroscope and acceleration data
            rotated_gyro_data = apply_rotation(gyro_data, rotation_matrix)
            rotated_acc_data = apply_rotation(acc_data, rotation_matrix)
            
            # Replace the original data with the rotated data
            df.iloc[:, 0:2] = df.iloc[:, 0:2].values
            df.iloc[:, 2:5] = rotated_gyro_data
            df.iloc[:, 5:8] = rotated_acc_data
            
            # Save the rotated CSV data
            # rotated_csv_path = os.path.join(save_directory, subdirectory, "imu_samples_0.csv")
            rotated_csv_path = os.path.join(save_directory, subdirectory, "imu_samples_0.csv")

            df.to_csv(rotated_csv_path, index=False)
    print(path)
    
    
for subdir in os.listdir(data_directory):
        subdir_path = os.path.join(data_directory, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Iterate through files in the subdirectory
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)

                # Check if it's a JSON or CSV file
                # if file.endswith(".json") or file.endswith(".csv"):
                if file.endswith(".json"):
                    # Create the destination subdirectory if it doesn't exist
                    destination_subdirectory = os.path.join(save_directory, subdir)
                    os.makedirs(destination_subdirectory, exist_ok=True)

                    # Copy the file to the destination subdirectory
                    shutil.copy(file_path, destination_subdirectory)

copy_files_from_other_list(data_directory, all_list_path, save_directory)

# List of specific files to copy
specific_files = ['all_ids.txt', 'spline_metrics.csv', 'test_list.txt', 'train_list.txt', 'val_list.txt', 'test_list1.txt', 'test_list2.txt', 'test_list3.txt', 'test_list4.txt']

# Copy the specific files from the source directory to the destination directory
for file in specific_files:
    file_path = os.path.join(data_directory, file)
    if os.path.exists(file_path):
        shutil.copy(file_path, save_directory)
        
# import os

# def count_files_in_subdirectories(directory):
#     total_files = 0

#     # Iterate through subdirectories
#     for subdir, _, files in os.walk(directory):
#         total_files += len(files)

#     return total_files

# # Example usage
# directory_path = "./local_data_test_so3/tlio_golden"

# total_files_count = count_files_in_subdirectories(directory_path)
# print(f"Total number of files in subdirectories: {total_files_count}")