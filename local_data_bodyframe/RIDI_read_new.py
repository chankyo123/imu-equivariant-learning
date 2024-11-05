import pandas as pd
import os
import numpy as np

def main():
    path_base = '/home/sangli/Downloads/DATASET/ridi-robust-imu-double-integration/versions/1/data_publish_v2'
    save_path = 'local_data_bodyframe/RIDI'

    seq_name = [f for f in os.listdir(path_base) if "body" in f]
    # print("seq_name: ", seq_name)
    for seq in seq_name:
        print("seq: ", seq)
        read_path = os.path.join(path_base, seq, "processed/data.csv")
        data_read = pd.read_csv(read_path)
        ts = data_read[['time']].values 
        gyro = data_read[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = data_read[['acce_x', 'acce_y', 'acce_z']].values
        tango_pos = data_read[['pos_x', 'pos_y', 'pos_z']].values
        tango_ori = data_read[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

        imu_save_path = os.path.join(save_path, seq, 'mav0', 'imu0', 'data.csv')
        imu_data = np.hstack((ts, gyro, acce))
        gt_save_path = os.path.join(save_path, seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        gt_data = np.hstack((ts, tango_pos, tango_ori))
        print("imu_data.shape: ", imu_data.shape)
        break


if __name__ == '__main__':
    main()