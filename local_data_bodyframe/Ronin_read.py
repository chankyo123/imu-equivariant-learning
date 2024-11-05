import os 
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scipy.signal import savgol_filter
from scipy.linalg import expm
from scipy.spatial.transform import Rotation 
import csv

def apply_sg_smoother(data, window_length, polyorder, deriv = 0):
        """
        Apply Savitzky-Golay filter to the data
        data: numpy array of shape (n,6)
        """
        dada_soothed = np.zeros_like(data)
        n = data.shape[-1]
        for i in range(n):
            dada_soothed[:, i] = savgol_filter(data[:, i], window_length, polyorder, deriv=deriv)
        return dada_soothed

def compute_velocity(gt_file_path):
    time_list = []
    p_list = []
    quat_list = []
    with open(gt_file_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        header = rows[0]
        data_rows = rows[1:]
        for row in data_rows: # row: [timestamp, px, py, pz, qw, qx, qy, qz]
            time_list.append(row[0])
            p_list.append(np.array([float(row[1]), float(row[2]), float(row[3])]))
            quat_list.append(np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])]))
    p_array = np.array(p_list) # shape: [N, 3]
    v_array = apply_sg_smoother(p_array, window_length=3, polyorder=1, deriv=1) / 0.005 # shape: [N, 3]
            
    # rewrite the gt file
    with open(gt_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header + ['vx', 'vy', 'vz'])
        for i in range(len(time_list)):
            writer.writerow([time_list[i]] + list(p_array[i]) + list(quat_list[i]) + list(v_array[i]))
    print("velocity computed and saved to: ", gt_file_path)

def plot_velocity(gt_file_path):
    import matplotlib.pyplot as plt
    data = np.loadtxt(gt_file_path, delimiter=',', skiprows=1)
    t = data[:, 0]
    t = (t - t[0]) / 1e9
    p = data[:, 1:4]
    v = data[:, 8:11]
    v_num_diff = (p[1:] - p[:-1]) / (t[1:] - t[:-1]).reshape(-1, 1)
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].plot(t, v[:, i], label='velocity_smoothed')
        ax[i].plot(t[:-1], v_num_diff[:, i], label='velocity_num_diff')
        ax[i].legend()
    fig.suptitle('velocity')
    fig.tight_layout()
    plt.show()

def skew_symmetric(w):
    """
    w: numpy array of shape (3,)
    """
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def gyro_integration(gyro, t):
    """
    gyro: numpy array of shape (N, 3)
    t: numpy array of shape (N, 1)
    """
    N = gyro.shape[0]
    Rot = np.zeros((N, 3, 3))
    for i in range(1, N):
        dt = t[i] - t[i-1]
        Rot[i] = Rot[i-1] @ expm(dt * skew_symmetric(gyro[i]))
    return Rot

def Rot2euler(R):
    """
    R: numpy array of shape (3, 3)
    """
    phi = np.arctan2(R[2, 1], R[2, 2])
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])

def main_old():
    # Read the data

    raw_data_path = ["/home/sangli/Downloads/DATASET/train_dataset_1", "/home/sangli/Downloads/DATASET/train_dataset_2"]
    for raw_data_path in raw_data_path:
        folder_name = [filename for filename in os.listdir(raw_data_path) if filename.startswith("a")]

        for index in range(len(folder_name)):
            print('Processing folder: ', folder_name[index])
            t = [0]
            acc     = [0, 0, 0]
            gyro    = [0, 0, 0]
            mag     = [0, 0, 0]
            ori     = [0, 0, 0, 0]
            pose    = [0, 0, 0] 

            load_file = os.path.join(raw_data_path, folder_name[index]) + '/data.hdf5'
            df = h5py.File(load_file, 'r')

            header = np.array(df.get('synced'))
            for i in range(len(np.array(df.get('synced')))):
                # np.array(df.get('synced'))[header[i]]
                if header[i] == 'time':
                    temp = np.array(df.get('synced')[header[i]]).reshape(-1, 1) * 1e9 # convert to ns
                    t = np.vstack((t, temp))
                if header[i] == 'acce':
                    temp = np.array(df.get('synced')[header[i]])
                    acc = np.vstack((acc, temp))
                if header[i] == 'gyro':
                    temp = np.array(df.get('synced')[header[i]])
                    gyro = np.vstack((gyro, temp))
                if header[i] == 'magnet':
                    temp = np.array(df.get('synced')[header[i]])
                    mag = np.vstack((mag, temp))
            header = np.array(df.get('pose'))
            for i in range(len(np.array(df.get('pose')))):
                if header[i] == 'ekf_ori':
                    temp = np.array(df.get('pose')[header[i]])
                    ori = np.vstack((ori, temp))
                if header[i] == 'tango_pos':
                    temp = np.array(df.get('pose')[header[i]])
                    pose = np.vstack((pose, temp))
            
            imu_data_save_path = os.path.join('local_data_bodyframe','Ronin', folder_name[index], 'mav0', 'imu0', 'data.csv')
            gt_data_save_path = os.path.join('local_data_bodyframe','Ronin', folder_name[index], 'mav0', 'state_groundtruth_estimate0', 'data.csv')
            os.makedirs(os.path.dirname(imu_data_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(gt_data_save_path), exist_ok=True)

            df_imu = pd.DataFrame({'time': t[1:,0], 'Acc x': acc[1:, 0], 'Acc y': acc[1:, 1], 'Acc z': acc[1:, 2], 'Gyro x': gyro[1:, 0],
                            'Gyro y': gyro[1:, 1], 'Gyro z': gyro[1:, 2], 'Mag x': mag[1:, 0], 'Mag y': mag[1:, 1], 'Mag z': mag[1:, 2]})
            df_imu.to_csv(imu_data_save_path, index=False)
            df_gt = pd.DataFrame({'time': t[1:,0],'Ori w': ori[1:, 0], 'Ori x': ori[1:, 1], 'Ori y': ori[1:, 2],
                                'Ori z': ori[1:, 3], 'Pose x': pose[1:, 0], 'Pose y': pose[1:, 1], 'Pose z': pose[1:, 2]})
            df_gt.to_csv(gt_data_save_path, index=False)
            print('IMU data saved to: ', imu_data_save_path)
            print('Ground truth data saved to: ', gt_data_save_path)

    seq_name = [f for f in os.listdir('local_data_bodyframe/Ronin') if os.path.isdir(os.path.join('local_data_bodyframe/Ronin', f))]
    for seq in seq_name:
        print("seq: ", seq)
        gt_file_path = os.path.join('local_data_bodyframe/Ronin', seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        compute_velocity(gt_file_path)
        plot_velocity(gt_file_path)
        # plot_position(gt_file_path)
def from_quaternion(quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        """
        if ordering == 'xyzw':
            qx = quat[:, 0]
            qy = quat[:, 1]
            qz = quat[:, 2]
            qw = quat[:, 3]
        elif ordering == 'wxyz':
            qw = quat[:, 0]
            qx = quat[:, 1]
            qy = quat[:, 2]
            qz = quat[:, 3]

        # Form the matrix
        mat = quat.new_empty(quat.shape[0], 3, 3)

        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
        mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
        mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

        mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
        mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
        mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

        mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
        mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
        mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)
        return mat

def imu_integration(imu_data, t, R0, v0, p0):
    """
    imu_data: numpy array of shape (N, 6) [wx, wy, wz, ax, ay, az]
    t: numpy array of shape (N, 1)
    """
    N = t.shape[0]
    Rot = np.zeros((N, 3, 3))
    v = np.zeros((N, 3))
    p = np.zeros((N, 3))
    if R0.shape[0] == N:
        Rot[0] = R0[0]
    else:
        Rot[0] = R0
    v[0] = v0
    p[0] = p0
    g_const = np.array([0, 0, -9.81])
    for i in range(1, N):
        dt = t[i] - t[i-1]
        if R0.shape[0] == N:
            Rot[i] = R0[i]
        else:
            Rot[i] = Rot[i-1] @ expm(dt * skew_symmetric(imu_data[i, :3]))
        v[i] = v[i-1] + dt * (Rot[i-1] @ imu_data[i, 3:] + g_const)
        p[i] = p[i-1] + dt * v[i-1] + 0.5 * dt**2 * (Rot[i-1] @ imu_data[i, 3:] + g_const)
    return Rot, v, p

def main():
    # Read the data
    path_base = "/home/sangli/Downloads/DATASET/ronin_train_all"
    seq_names = [f for f in os.listdir(path_base) if os.path.isdir(os.path.join(path_base, f))]
    save_path_base = 'local_data_bodyframe/Ronin'
    seq_names = ["a002_1"]
    for seq in seq_names:
        print("seq: ", seq)
        imu_data_path = os.path.join(path_base, seq, 'imu_data.txt')
        gt_data_path = os.path.join(path_base, seq, 'gt_data.txt')
        imu_data = np.loadtxt(imu_data_path, delimiter=',', skiprows=1)
        gt_data = np.loadtxt(gt_data_path, delimiter=',', skiprows=1)
        imu_data[:,0] = imu_data[:,0] * 1e9
        gt_data[:,0] = gt_data[:,0] * 1e9
        imu_data_save_path = os.path.join(save_path_base, seq, 'mav0', 'imu0', 'data.csv')
        gt_data_save_path = os.path.join(save_path_base, seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        os.makedirs(os.path.dirname(imu_data_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(gt_data_save_path), exist_ok=True)
        np.savetxt(imu_data_save_path, imu_data, delimiter=',', header='time(ns), wx, wy, wz, ax, ay, az)')
        np.savetxt(gt_data_save_path, gt_data, delimiter=',', header='time(ns), px, py, pz, qw, qx, qy, qz')
        print('IMU data saved to: ', imu_data_save_path)
        print('Ground truth data saved to: ', gt_data_save_path)

    seq_name = [f for f in os.listdir(save_path_base) if os.path.isdir(os.path.join(save_path_base, f))]
    seq_name = ["a002_1"]
    for seq in seq_name:
        print("seq: ", seq)
        gt_file_path = os.path.join(save_path_base, seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        compute_velocity(gt_file_path)
        # plot_velocity(gt_file_path)
        # check 
        imu = np.loadtxt(os.path.join(save_path_base, seq, 'mav0', 'imu0', 'data.csv'), delimiter=',', skiprows=1)
        gt = np.loadtxt(gt_file_path, delimiter=',', skiprows=1)
        t = imu[:, 0] / 1e9
        t = t - t[0]
        imu_data = imu[:, 1:]
        R_ALL = from_quaternion(torch.from_numpy(gt[:, 4:8]), ordering='wxyz').numpy()
        v_ALL = gt[:, 8:11]
        p_ALL = gt[:, 1:4]
        NN = 20
        t = t[:NN]
        Rot, v, p = imu_integration(imu_data, t, R_ALL[0], v_ALL[0], p_ALL[0])
        Rot_int = Rotation.from_matrix(Rot)
        Rot_gt = Rotation.from_quat(gt[:NN, 4:8],scalar_first=True)
        Rot_int_euler = Rot_int.as_euler('xyz', degrees=True)
        Rot_gt_euler = Rot_gt.as_euler('xyz', degrees=True)
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(t, Rot_int_euler[:, i], label='Rot_int')
            ax[i].plot(t, Rot_gt_euler[:, i], label='Rot_gt')
            ax[i].legend()
        fig.suptitle('Rot')
        fig.tight_layout()

        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(t, v[:, i], label='velocity_integrated')
            ax[i].plot(t, v_ALL[:NN, i], label='velocity_gt')
            ax[i].legend()
        fig.suptitle('velocity')
        fig.tight_layout()
        plt.show()
    


if __name__ == "__main__":
    main()