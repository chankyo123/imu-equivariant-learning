import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Load the npy file
# file_path = "sim_imu_longerseq/10/imu0_resampled.npy" 
# file_path = "sim_imu_longerseq_worldframe/17/imu0_resampled.npy" 
# file_path = "sim_imu_longerseq_idso2/17/imu0_resampled.npy" 
file_path = "local_data_bodyframe/tlio_golden/145820422949970/imu0_resampled.npy" 
file_path = "sim_imu_longerseq_may/1/imu0_resampled_fromcsv.npy" 
# file_path = "local_data_test_so2_2023/tlio_golden/145820422949970/imu0_resampled.npy" 

# file_path = "sim_imu_longerseq/1/imu0_resampled.npy" 
# data = np.load(file_path)
# print(data[:10,0])
# file_path = "june_sim_imu_longerseq/1/imu0_resampled.npy" 
# data = np.load(file_path)
# print(data[:10,0])
# # file_path = "local_data_test_so3_fixed_2/tlio_golden/145820422949970/imu0_resampled.npy" 
# file_path = "./so3_local_bata_bodyframe2/182009952689275/imu0_resampled.npy" 
# data = np.load(file_path)
# print(data[:10,0])
# assert False
# Extract position data (pos_World_Device)
# Assuming position data is in columns 11, 12, 13 (0-indexed)
def plot_xy_traj(data):
    pos_x = data[:, 11]
    pos_y = data[:, 12]
    pos_z = data[:, 13]

    # Plot the 2D trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(pos_x, pos_y, label='Trajectory')
    # plt.plot(pos_y, pos_z, label='Trajectory')
    plt.xlabel('Position X (World Frame)')
    plt.ylabel('Position Y (World Frame)')
    plt.title('2D Trajectory Plot (X-Y)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling on both axes
    # plt.show()
    plt.savefig("a_traj_plot.png")
    
def acc_plot(data):
    time = data[:,0]

    # Extract accelerometer components
    acc_x = data[:, 4]
    acc_y = data[:, 5]
    acc_z = data[:, 6]

    vel_b_x = data[:, 14]
    vel_b_y = data[:, 15]
    vel_b_z = data[:, 16]

    # Plot accelerometer data
    plt.figure(figsize=(10, 6))
    # plt.plot(time, acc_x, label='Acc X')
    # plt.plot(time, acc_y, label='Acc Y')
    # plt.plot(time, acc_z, label='Acc Z')
    plt.plot(time, vel_b_x, label='Vel B X')
    plt.plot(time, vel_b_y, label='Vel B Y')
    plt.plot(time, vel_b_z, label='Vel B Z')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Accelerometer Data')
    plt.legend()
    plt.grid(True)
    plt.savefig("a_velplot.png")

def compute_and_plot_trajectory(npy_file_path, save_file_path=None):
    """
    Compute positions from velocities using cumulative sum and plot the 2D XY trajectory.

    Parameters:
    npy_file_path (str): Path to the input .npy file containing the data.
    save_file_path (str, optional): Path to save the computed positions. Defaults to None.

    Returns:
    None
    """
    # Load the .npy file
    data = np.load(npy_file_path)

    # Extract time and velocity components
    time = data[:, 0]
    vel_b_x = data[:, -3]
    vel_b_y = data[:, -2]
    vel_b_z = data[:, -1]

    # Calculate position by taking the cumulative sum of the velocity components

    # Assuming the time step is the difference between consecutive time points
    time_step = np.diff(time)
    time_step = np.append(time_step, time_step[-1])  # Append the last time step to match the length

    # Integrate velocity to get position
    pos_b_x = np.cumsum(vel_b_x * time_step)
    pos_b_y = np.cumsum(vel_b_y * time_step)
    pos_b_z = np.cumsum(vel_b_z * time_step)

    # Combine the positions into a single array
    positions = np.vstack((pos_b_x, pos_b_y, pos_b_z)).T

    # Save the positions to a new .npy file if a save path is provided
    if save_file_path:
        np.save(save_file_path, positions)

    # Plot the 2D XY trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(positions[:, 0], positions[:, 1], label='2D XY Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D XY Trajectory')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def compute_and_plot_trajectory_with_acceleration(npy_file_path, save_file_path=None):
    # Load the .npy file
    data = np.load(npy_file_path)
    # data[:,6] = -data[:,6]
    # np.save("sim_imu_longerseq/17/imu0_resampled1.npy", data)
    # return None

    # Extract time and velocity components
    time = data[:, 0]
    vel_b_x = data[:, -3]
    vel_b_y = data[:, -2]
    vel_b_z = data[:, -1]

    # Extract accelerometer components from the .npy file
    acc_x = data[:, 4]
    acc_y = data[:, 5]
    acc_z = data[:, 6]

    time_step = np.diff(time) * 1e-6
    time_step = np.append(time_step, time_step[-1])  # Append the last time step to match the length

    # Calculate numerical accelerations by differentiating the velocity components
    rotation_info = np.load("./so3_local_data_bodyframe2/rpy_values.npy")
    rotation_rpy = rotation_info[0,:]
    m_b2bprime = Rotation.from_euler('xyz', rotation_rpy, degrees=False).as_matrix()
    # m_b2bprime = np.eye(3)
    
    gravity_vector = m_b2bprime @ np.array([0, 9.81, 0])
    acc_calc_x = np.diff(vel_b_x) / time_step[:-1] + gravity_vector[0]
    acc_calc_y = np.diff(vel_b_y) / time_step[:-1] + gravity_vector[1]
    acc_calc_z = np.diff(vel_b_z) / time_step[:-1] + gravity_vector[2]

    # Append the last acceleration value to maintain the same length
    acc_calc_x = np.append(acc_calc_x, acc_calc_x[-1])
    acc_calc_y = np.append(acc_calc_y, acc_calc_y[-1])
    acc_calc_z = np.append(acc_calc_z, acc_calc_z[-1])

    # Plot the accelerations
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    
    # Plot accelerations from .npy file
    axs[0, 0].plot(time[:2000], acc_x[:2000], label='ACC X (npy)')
    axs[0, 0].set_title('ACC X (npy)')
    axs[1, 0].plot(time[:2000], acc_y[:2000], label='ACC Y (npy)')
    axs[1, 0].set_title('ACC Y (npy)')
    axs[2, 0].plot(time[:2000], acc_z[:2000], label='ACC Z (npy)')
    axs[2, 0].set_title('ACC Z (npy)')
    
    # Plot calculated accelerations
    axs[0, 1].plot(time[:2000], acc_calc_x[:2000], label='ACC X (calc)', color='r')
    axs[0, 1].set_title('ACC X (calc)')
    axs[1, 1].plot(time[:2000], acc_calc_y[:2000], label='ACC Y (calc)', color='r')
    axs[1, 1].set_title('ACC Y (calc)')
    axs[2, 1].plot(time[:2000], acc_calc_z[:2000], label='ACC Z (calc)', color='r')
    axs[2, 1].set_title('ACC Z (calc)')

    # Set labels
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Acceleration')
        ax.legend()
        ax.grid()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("acc-compare.png")
    
    
def load_and_process_worldframe_acc(file_path):
    """
    Load IMU data from a .npy file, compensate for gravity, integrate accelerations and velocities,
    and plot the 2D XY trajectory.

    Parameters:
    file_path (str): Path to the input .npy file containing the data.

    Returns:
    None
    """
    # Load the .npy file
    data = np.load(file_path)
    # data[:, 4:7] = -data[:, 4:7]
    # data[:, 4:7] = data[:, 4:7] + np.array([0, 0, 2*9.81])
    # np.save("sim_imu_longerseq/17/imu0_resampled2.npy", data)
    # print("saved")
    # return None
    # Extract components
    time = data[:, 0]
    acc_world = data[:, 4:7]  # Acceleration in the world frame
    gt_vel = data[:, -3:] 

    # Gravity vector in world frame (assuming gravity is -9.81 m/s^2 in Z direction)
    gravity = np.array([0, 0, -9.81])

    # Compensate for gravity
    acc_compensated = acc_world + gravity

    # Calculate time step
    time_step = np.diff(time)*1e-6
    time_step = np.append(time_step, time_step[-1])  # Append the last time step to match the length

    # Integrate acceleration to get velocity
    vel = np.cumsum(acc_compensated * time_step[:, None], axis=0)+gt_vel[0:1,:]
    # print(gt_vel - vel)
    # print(gt_vel[:10,:])
    # print(vel[:10,:])
    # Integrate velocity to get position
    pos = np.cumsum(vel * time_step[:, None], axis=0)
    pos_gt = np.cumsum(gt_vel * time_step[:, None], axis=0)
    print(pos_gt[:10,:])
    print(pos[:10,:])
    # # Plot velocities and positions in subplots
    fig, axs = plt.subplots(2, figsize=(15, 10))

    # # Plot velocity
    # axs[0, 0].plot(time, vel[:, 0], label='Velocity X')
    # axs[0, 0].plot(time, vel[:, 1], label='Velocity Y')
    # axs[0, 0].plot(time, vel[:, 2], label='Velocity Z')
    # axs[0, 0].set_title('Velocity')
    # axs[0, 0].set_xlabel('Time')
    # axs[0, 0].set_ylabel('Velocity (m/s)')
    # axs[0, 0].legend()
    # axs[0, 0].grid()

    # # Plot position
    # axs[0, 1].plot(time, pos[:, 0], label='Position X')
    # axs[0, 1].plot(time, pos[:, 1], label='Position Y')
    # axs[0, 1].plot(time, pos[:, 2], label='Position Z')
    # axs[0, 1].set_title('Position')
    # axs[0, 1].set_xlabel('Time')
    # axs[0, 1].set_ylabel('Position (m)')
    # axs[0, 1].legend()
    # axs[0, 1].grid()

    # # Plot 2D XY trajectory from positions
    axs[0].plot(pos[:100000, 0], pos[:100000, 1], label='2D XY Trajectory')
    axs[0].set_title('2D XY Trajectory')
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    axs[0].legend()
    axs[0].grid()
    axs[0].axis('equal')
    
    # axs[1].plot(data[:, -3], data[:, -2], label='2D XY Trajectory')
    axs[1].plot(pos_gt[:100000, 0], pos_gt[:100000, 1], label='2D XY Trajectory')
    axs[1].set_title('2D XY Trajectory')
    axs[1].set_xlabel('X Position')
    axs[1].set_ylabel('Y Position')
    axs[1].legend()
    axs[1].grid()
    axs[1].axis('equal')
    plt.savefig("traj-world.png")
    
    # # Adjust layout
    # plt.tight_layout()
    # plt.show()
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Plot cumulative sum of the world velocity
    axs[0, 0].plot(time, vel[:, 0], label='Cumulative Velocity X')
    axs[1, 0].plot(time, vel[:, 1], label='Cumulative Velocity Y')
    axs[2, 0].plot(time, vel[:, 2], label='Cumulative Velocity Z')
    axs[0, 0].set_title('Cumulative Velocity X')
    axs[1, 0].set_title('Cumulative Velocity Y')
    axs[2, 0].set_title('Cumulative Velocity Z')
    
    # Plot ground truth velocity
    axs[0, 1].plot(time, gt_vel[:, 0], label='Ground Truth Velocity X', color='r')
    axs[1, 1].plot(time, gt_vel[:, 1], label='Ground Truth Velocity Y', color='r')
    axs[2, 1].plot(time, gt_vel[:, 2], label='Ground Truth Velocity Z', color='r')
    axs[0, 1].set_title('Ground Truth Velocity X')
    axs[1, 1].set_title('Ground Truth Velocity Y')
    axs[2, 1].set_title('Ground Truth Velocity Z')
    
    # Set labels and legends
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Velocity (m/s)')
        ax.legend()
        ax.grid()

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig("velplot-world.png")
    
def get_acc(file_path):
    """
    Load IMU data from a .npy file, use provided compensated accelerations, calculate accelerations
    from the difference of ground truth velocities, and plot the comparison of calculated and ground truth
    accelerations for each axis.

    Parameters:
    file_path (str): Path to the input .npy file containing the data.

    Returns:
    None
    """
    # Load the .npy file
    data = np.load(file_path)
    
    # Extract components
    time = data[:, 0]
    acc_compensated = data[:, 4:7]  # Compensated acceleration in the world frame
    # gravity = np.array([0, 0, -9.81])
    # acc_compensated = acc_world + gravity
    gt_vel = data[:, -3:]           # Ground truth velocity in the world frame

    # Calculate time step
    time_step = np.diff(time)
    time_step = np.append(time_step, time_step[-1])  # Append the last time step to match the length

    # Calculate ground truth acceleration from velocity
    gt_acc = np.diff(gt_vel, axis=0) / time_step[:-1, None]
    gt_acc = np.vstack([gt_acc, gt_acc[-1]])  # Append the last acceleration to match the length

    # Plot accelerations in subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Plot calculated (compensated) acceleration
    axs[0, 0].plot(time, acc_compensated[:, 0], label='Calculated Acceleration X')
    axs[1, 0].plot(time, acc_compensated[:, 1], label='Calculated Acceleration Y')
    axs[2, 0].plot(time, acc_compensated[:, 2], label='Calculated Acceleration Z')
    axs[0, 0].set_title('Calculated Acceleration X')
    axs[1, 0].set_title('Calculated Acceleration Y')
    axs[2, 0].set_title('Calculated Acceleration Z')
    
    # Plot ground truth acceleration
    axs[0, 1].plot(time, gt_acc[:, 0], label='Ground Truth Acceleration X', color='r')
    axs[1, 1].plot(time, gt_acc[:, 1], label='Ground Truth Acceleration Y', color='r')
    axs[2, 1].plot(time, gt_acc[:, 2], label='Ground Truth Acceleration Z', color='r')
    axs[0, 1].set_title('Ground Truth Acceleration X')
    axs[1, 1].set_title('Ground Truth Acceleration Y')
    axs[2, 1].set_title('Ground Truth Acceleration Z')
    
    # Set labels and legends
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Acceleration (m/s^2)')
        ax.legend()
        ax.grid()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("acceleration_comparison.png")

def process_imu_data(npy_path):
    """
    Load IMU data, compensate for gravity, and integrate to obtain world velocity and pose.

    Parameters:
    npy_path (str): Path to the .npy file containing IMU data.

    Returns:
    tuple: world velocity and world pose arrays.
    """
    # Load the .npy file
    data = np.load(npy_path)

    # Extract time, body-frame accelerations, and quaternions from data
    time = data[:, 0]
    acc_body = data[:, 4:7]  # Adjust indices based on your data format
    quaternions = data[:, -10:-6]
    

    # Calculate time steps (dt) from the time column
    dt = np.diff(time) * 1e-6
    dt = np.insert(dt, 0, dt[0])  # Insert the first time step to maintain array length
    # Convert quaternions to rotation matrices
    rotation_matrices = [Rotation.from_quat(quat).as_matrix() for quat in quaternions]

    # Compensate for gravity
    rotation_info = np.load("./so3_local_data_bodyframe2/rpy_values.npy")
    rotation_rpy = rotation_info[0,:]
    m_b2bprime = Rotation.from_euler('xyz', rotation_rpy, degrees=False).as_matrix()
    m_b2bprime = np.eye(3)
    gravity_vector = m_b2bprime @ np.array([0, 0, -9.81])
    acc_world = []
    for i in range(len(acc_body)):
        R_b2w = rotation_matrices[i]
        acc_world_i = R_b2w @ acc_body[i] + m_b2bprime @ gravity_vector
        acc_world.append(acc_world_i)
    acc_world = np.array(acc_world)

    # Integrate to get world velocity and pose
    velocity_world = np.zeros_like(acc_world)
    velocity_gt = np.zeros_like(acc_world)
    pose_world = np.zeros((acc_world.shape[0], 3))  # Assuming 3D position
    for i in range(1, len(acc_world)):
        velocity_world[i] = velocity_world[i - 1] + acc_world[i] * dt[i]
        R_b2w = rotation_matrices[i]
        velocity_gt[i] = R_b2w @ data[i, -3:]
        # if i < 100:
        #     print(data[i, -6:-3])
        pose_world[i] = pose_world[i - 1] + velocity_world[i] * dt[i]
        if i < 1000:
            # print(pose_world[i])
            print(velocity_world[i]-velocity_gt[i])
        

    # Plot and save the 2D XY trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(pose_world[:, 0], pose_world[:, 1], label='Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D XY Trajectory')
    plt.legend()
    plt.grid(True)
    plt.savefig("pose_integration_from_acc.png")
    return 


    
# npy_file_path = 'sim_imu_longerseq_worldframe/17/imu0_resampled.npy'
# npy_file_path = 'local_data/tlio_golden/145820422949970/imu0_resampled.npy'
# npy_file_path = 'sim_imu_longerseq_worldframe/25/imu0_resampled.npy'
# npy_file_path = 'local_data_test_so3/tlio_golden/145820422949970/imu0_resampled.npy'
# npy_file_path = 'local_data_test_so3_fixed_2/tlio_golden/145820422949970/imu0_resampled.npy'

npy_file_path = 'so3_local_data_bodyframe2/145820422949970/imu0_resampled.npy'
npy_file_path = 'local_data/tlio_golden/145820422949970/imu0_resampled.npy'

# npy_file_path = 'sim_imu_longerseq/17/imu0_resampled.npy'
save_file_path = None

# plot_xy_traj(data)
# acc_plot(data)
# compute_and_plot_trajectory(npy_file_path, save_file_path)
# compute_and_plot_trajectory_with_acceleration(npy_file_path, save_file_path)

# process_imu_data(npy_file_path)

load_and_process_worldframe_acc(npy_file_path)
# get_acc(npy_file_path)


# import numpy as np

# def compare_npy_files(file_path1, file_path2):
#     # Load the data from both npy files
#     data1 = np.load(file_path1)
#     data2 = np.load(file_path2)
    
#     # Check if shapes of both arrays are the same
#     if data1.shape != data2.shape:
#         print("The files have different shapes.")
#         return False
    
#     # Check if the data in both arrays are the same
#     if np.array_equal(data1, data2):
#         print("The files are identical.")
#         return True
#     else:
#         print("The files are not identical.")
#         return False

# # File paths for the npy files
# file_path1 = "sim_imu_longerseq_worldframe_idso2/100/imu0_resampled.npy" 
# file_path2 = "sim_imu_longerseq_idso2_2_worldframe/100/imu0_resampled.npy"

# # Compare the files
# compare_npy_files(file_path1, file_path2)
