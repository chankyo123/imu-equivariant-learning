import numpy as np
import shutil
import os
import json
import yaml
import torch
import math
from scipy.spatial.transform import Rotation


data_list =[]
with open('local_data/tlio_golden/test_list_original.txt') as f:
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
        angle = torch.randn((1)) * math.pi
        R = torch.Tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0,0,1]]).numpy()
        P = torch.Tensor([[0,1,0],[1,0,0],[0,0,1]]).numpy()
        if os.path.isdir(path+folder+'_ref_rdn_aug'+str(0+idx)):
            print(path+folder+'_ref_rdn_aug'+str(0+idx)+'folder exists')
        else:
            os.mkdir(path+folder+'_ref_rdn_aug'+str(0+idx))
        aug_file_list.append(folder+'_ref_rdn_aug'+str(0+idx))
        
        ## copy imu0_resampled_description.npy
        shutil.copy(path+folder+'/imu0_resampled_description.json',path+folder+'_ref_rdn_aug'+str(0+idx)+'/imu0_resampled_description.json')

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

            ## apply the reflection on p,v,a,w
            gyro = -1 * np.einsum('ji,tj->ti', P,gyro)
            accel = np.einsum('ji,tj->ti',P,accel)
            pos = np.einsum('ji,tj->ti',P,pos)
            vel = np.einsum('ji,tj->ti',P,vel)

            np.save(path+folder+'_ref_rdn_aug'+str(0+idx)+'/imu0_resampled.npy', np.concatenate([ts.reshape((-1,1)), gyro, accel, q_ori, pos, vel], axis=1))
        
        ### calibration file
       ## copy because not used in neural network
        shutil.copy(path+folder+'/calibration.json',path+folder+'_ref_rdn_aug'+str(0+idx)+'/calibration.json')
     

print(len(aug_file_list))
with open(path +'test_list_ref_rdn_aug.txt', "w") as output:
    for file in aug_file_list:
        output.write(file+'\n')

    


