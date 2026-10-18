import numpy as np
import shutil
import os
import json
import yaml
import torch
import math
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d


data_list =[]
with open('local_data/tlio_golden/test_list.txt') as f:
    data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
path = 'local_data/tlio_golden/'


n_rot = [4,8,16,24,32]
angle_list = list(np.linspace(0,360,n_rot[0],endpoint=False))
for idx in range(len(n_rot)):
    print(len(data_list))
    count = 0
    aug_file_list = [] 
    print(angle_list)
    for rot in angle_list:
        print(rot)
        for folder in data_list:
            count += 1
            # print(folder) 
            angle = torch.Tensor([rot/(2*math.pi)])
            # if (count+idx)%2==0: ## rotations
            R = torch.Tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0,0,1]]).numpy()
            # else: ##reflections
            #     R = torch.Tensor([[-torch.sin(angle), torch.cos(angle), 0], [torch.cos(angle), torch.sin(angle), 0],[0,0,1]]).numpy()
            if os.path.isdir(path+folder+'_aug'+str(int(rot))+str(0+idx)):
                print(path+folder+'_aug'+str(int(rot))+str(0+idx)+'folder exists')
            else:
                os.mkdir(path+folder+'_aug'+str(int(rot))+str(0+idx))
            aug_file_list.append(folder+'_aug'+str(int(rot))+str(0+idx))
            
            ## copy imu0_resampled_description.npy
            shutil.copy(path+folder+'/imu0_resampled_description.json',path+folder+'_aug'+str(int(rot))+str(0+idx)+'/imu0_resampled_description.json')

            ## read imu0_resampled.npy
            if os.path.exists(path+folder+'/imu0_resampled.npy'):
                data = np.load(path+folder+'/imu0_resampled.npy')
                ts = data[:,0]
                gyro = data[:,1:4]
                accel = data[:,4:7]
                q_ori = data[:,7:11] # need to apply rotation only to orientation - post multiplication
                pos = data[:,11:14]
                vel = data[:, 14:] 

                R_ori = Rotation.from_quat(q_ori).as_matrix()
                R_ori = np.einsum('tij,jk -> tik',R_ori, R)
                q_ori = Rotation.from_matrix(R_ori).as_quat()


                np.save(path+folder+'_aug'+str(int(rot))+str(0+idx)+'/imu0_resampled.npy', np.concatenate([ts.reshape((-1,1)), gyro, accel, q_ori, pos, vel], axis=1))
            
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

            with open(path+folder+'_aug'+str(int(rot))+str(0+idx)+'/calibration.json', "w") as outfile: 
                json.dump(calib_json, outfile)          

            


    print(len(aug_file_list))
    with open(path +'test_list_aug'+str(idx)+'.txt', "w") as output:
        for file in aug_file_list:
            output.write(file+'\n')

    angle_list = [a for a in list(np.linspace(0,360,n_rot[idx+1],endpoint=False)) if a not in list(np.linspace(0,360,n_rot[idx],endpoint=False))]
    

## split the test datasets into separate files to parallel process
for i in range(5):
    data_list =[]
    with open(path +'test_list_aug'+str(i)+'.txt') as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
        if len(data_list)>500:
            for j in range(0,len(data_list),500):
                with open(path +'test_list_aug'+str(i)+'part'+str(j)+'.txt', "w") as output:
                    end_j = j+500
                    if end_j > len(data_list):
                        end_j = len(data_list)
                    for file in data_list[j:end_j]:
                        output.write(file+'\n')
