from pathlib import Path
from rosbags.highlevel import AnyReader

import tarfile
import os
import csv
import numpy as np
from scipy.signal import savgol_filter

def read_IMU_from_txt(txt_file_path, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ang_vel_txt_path = os.path.join(txt_file_path, 'gyro.txt')
    acc_txt_path = os.path.join(txt_file_path, 'acce.txt')
    imu_w_data = np.loadtxt(ang_vel_txt_path, delimiter=' ', skiprows=1)
    imu_a_data = np.loadtxt(acc_txt_path, delimiter=' ', skiprows=1)
    assert np.allclose(imu_w_data[:, 0], imu_a_data[:, 0])
    imu_data = np.concatenate([imu_w_data[:, 0].reshape(-1, 1), imu_w_data[:, 1:], imu_a_data[:, 1:]], axis=1)
    np.savetxt(output_file, imu_data, delimiter=',', header='timestamp,wx,wy,wz,ax,ay,az', comments='')#, fmt='%.6f')
    print("imu file saved to: ", output_file)

def read_GT_from_txt(gt_txt_file_path, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    gt_txt_path = os.path.join(gt_txt_file_path, 'pose.txt')
    gt_data = np.loadtxt(gt_txt_path, delimiter=' ', skiprows=1)
    np.savetxt(output_file, gt_data, delimiter=',', header='timestamp,px,py,pz,qw,qx,qy,qz', comments='')#, fmt='%.6f')
    print("gt file saved to: ", output_file)


def read_Tum_data(Tum_tar_path, imu_file_path, gt_file_path):
    os.makedirs(os.path.dirname(imu_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_file_path), exist_ok=True)
    success = 0
    with tarfile.open(Tum_tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('mocap0/data.csv'):
                extracted_file = tar.extractfile(member)
                with open(gt_file_path, 'wb') as output_file:
                    output_file.write(extracted_file.read())
                print("gt file saved to: ", gt_file_path)
                success += 1
            if member.name.endswith('imu0/data.csv'):
                extracted_file = tar.extractfile(member)
                with open(imu_file_path, 'wb') as output_file:
                    output_file.write(extracted_file.read())
                print("imu file saved to: ", imu_file_path)
                success += 1
            if success == 2:
                return
    print("file not found!")

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
    v_array = apply_sg_smoother(p_array, window_length=35, polyorder=5, deriv=1) / 0.005 # shape: [N, 3]
            
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

def plot_position(gt_file_path):
    import matplotlib.pyplot as plt
    data = np.loadtxt(gt_file_path, delimiter=',', skiprows=1)
    t = data[:, 0]
    t = (t - t[0]) / 1e9
    p = data[:, 1:4]
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].plot(t, p[:, i], label='position')
        ax[i].legend()
    fig.suptitle('position')
    fig.tight_layout()
    plt.show()

def main(): 
    ridi_data_path = '/home/sangli/Downloads/DATASET/ridi-robust-imu-double-integration/versions/1/data_publish_v2'
    save_path = 'local_data_bodyframe/RIDI'
    
    seq_name = [f for f in os.listdir(ridi_data_path) if "body" in f]
    print("seq_name: ", seq_name)
    for seq in seq_name:
        print("seq: ", seq)
        read_path = os.path.join(ridi_data_path, seq)
        output_file = os.path.join(save_path, seq, 'mav0', 'imu0', 'data.csv')
        read_IMU_from_txt(read_path, output_file)
        gt_save_path = os.path.join(save_path, seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        read_GT_from_txt(read_path, gt_save_path)

    seq_name = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]
    for seq in seq_name:
        print("seq: ", seq)
        gt_file_path = os.path.join(save_path, seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        compute_velocity(gt_file_path)
        plot_velocity(gt_file_path)
        # plot_position(gt_file_path)

    print("All done!")

if __name__ == '__main__':
    main()
    