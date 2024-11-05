from pathlib import Path
from rosbags.highlevel import AnyReader

import tarfile
import os
import csv
import numpy as np
from scipy.signal import savgol_filter

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
    v_array = apply_sg_smoother(p_array, window_length=11, polyorder=3, deriv=1) / 0.005 # shape: [N, 3]
            
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
    Tum_data_path = '/home/sangli/Downloads/DATASET'
    save_path = 'local_data_bodyframe/TUM'
    
    seq_name = [f for f in os.listdir(Tum_data_path) if f.endswith('.tar')]
    for seq in seq_name:
        print("seq: ", seq)
        Tum_tar_path = os.path.join(Tum_data_path, seq)
        imu_file_path = os.path.join(save_path, seq[:-4], 'mav0','imu0', 'data.csv')
        gt_file_path = os.path.join(save_path, seq[:-4], 'mav0','state_groundtruth_estimate0', 'data.csv')
        read_Tum_data(Tum_tar_path=Tum_tar_path, imu_file_path=imu_file_path, gt_file_path=gt_file_path)

    seq_name = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]
    for seq in seq_name:
        print("seq: ", seq)
        gt_file_path = os.path.join(save_path, seq, 'mav0','state_groundtruth_estimate0', 'data.csv')
        compute_velocity(gt_file_path)
        # plot_velocity(gt_file_path)
        # plot_position(gt_file_path)

    # print("seq_name: ", seq_name)
    # seq_test = 'local_data_bodyframe/TUM/dataset-corridor3_512_16/mav0/state_groundtruth_estimate0/data.csv'
    # # compute_velocity(seq_test)
    # # plot_velocity(seq_test)
    # # plot_position(seq_test)

    print("All done!")

if __name__ == '__main__':
    main()
    