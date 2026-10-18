import numpy as np
import shutil
import os
import json
import yaml
import torch
import math
from scipy.spatial.transform import Rotation


data_list =[]
with open('local_data/tlio_golden/test_list_orig.txt') as f:
    data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
path = 'local_data/tlio_golden/'
print(len(data_list))
count = 0
aug_file_list = []
for idx in range(4):
    print(idx)
    for folder in data_list: #only rotations
        count += 1
        # print(folder)
        angles = torch.randn((3)) * math.pi
        R = Rotation.from_euler('xyz',angles,degrees=False).as_matrix()
        if os.path.isdir(path+folder+'_rot_ekf_aug'+str(0+idx)):
            print(path+folder+'_rot_ekf_aug'+str(0+idx)+'folder exists')
        else:
            os.mkdir(path+folder+'_rot_ekf_aug'+str(0+idx))
        aug_file_list.append(folder+'_rot_ekf_aug'+str(0+idx))
        
        ## copy imu0_resampled_description.npy
        shutil.copy(path+folder+'/imu0_resampled_description.json',path+folder+'_rot_ekf_aug'+str(0+idx)+'/imu0_resampled_description.json')

        ## read imu0_resampled.npy
        if os.path.exists(path+folder+'/imu0_resampled.npy'):
            data = np.load(path+folder+'/imu0_resampled.npy')
            ts = data[:,0]
            gyro = data[:,1:4]
            accel = data[:,4:7]
            q_ori = data[:,7:11] # only orientation changes
            pos = data[:,11:14]
            vel = data[:, 14:] 

            R_ori = Rotation.from_quat(q_ori).as_matrix()
            R_ori = np.einsum('tij,jk -> tik',R_ori, R)
            q_ori = Rotation.from_matrix(R_ori).as_quat()

            np.save(path+folder+'_rot_ekf_aug'+str(0+idx)+'/imu0_resampled.npy', np.concatenate([ts.reshape((-1,1)), gyro, accel, q_ori, pos, vel], axis=1))
        
        ### calibration file
        with open(path+folder+'/calibration.json', 'r') as f:
            calib_json = json.load(f)
        accelBias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"])[:,None]
        gyroBias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"])[:,None]
        accelScaleInv = np.linalg.inv(np.array(
            calib_json["Accelerometer"]["Model"]["RectificationMatrix"]
        ))
        gyroScaleInv = np.linalg.inv(np.array(
            calib_json["Gyroscope"]["Model"]["RectificationMatrix"]
        ))
        ## All of this data is in inertial frame - so we need to apply R.T to them
        accelBias = np.einsum('ji,jk->ik',R,accelBias)
        gyroBias = np.einsum('ji,jk->ik',R,gyroBias)
        ## For rectification matrix it is R.T(GR) when G is in inertial frame
        accelScaleInv = np.einsum('ji,jk->ik',R,np.einsum('ij,jk->ik',accelScaleInv,R))
        gyroScaleInv = np.einsum('ji,jk->ik',R,np.einsum('ij,jk->ik',gyroScaleInv,R))
        calib_json["Accelerometer"]["Bias"]["Offset"]  = accelBias.reshape((-1)).tolist()
        calib_json["Gyroscope"]["Bias"]["Offset"] = gyroBias.reshape((-1)).tolist()
        calib_json["Accelerometer"]["Model"]["RectificationMatrix"] = np.linalg.inv(accelScaleInv).tolist()
        calib_json["Gyroscope"]["Model"]["RectificationMatrix"] = np.linalg.inv(gyroScaleInv).tolist() 

        with open(path+folder+'_rot_ekf_aug'+str(0+idx)+'/calibration.json', "w") as outfile: 
            json.dump(calib_json, outfile)       

        ##read imu_samples_0.csv 
        if os.path.exists(path+folder+'/imu_samples_0.csv'):
            data = np.loadtxt(path+folder+'/imu_samples_0.csv', delimiter=',')

            ts = data[:,0]
            temp = data[:,1]
            gyro = data[:,2:5]
            accel = data[:,-3:]

            ## All of this data is in inertial frame

            gyro = np.einsum('ji,tj->ti',R,gyro)
            accel = np.einsum('ji,tj->ti',R,accel)

            np.savetxt(path+folder+'_rot_ekf_aug'+str(0+idx)+'/imu_samples_0.csv', np.concatenate([ts.reshape((-1,1)), temp.reshape((-1,1)), gyro, accel], axis=1), delimiter=",")
    



print(len(aug_file_list))
with open(path +'test_list_rot_ekf_aug.txt', "w") as output:
    for file in aug_file_list:
        output.write(file+'\n')

    


