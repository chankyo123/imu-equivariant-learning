import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    # print(x.shape)     #torch.Size([1024, 1, 6, 200])
    batch_size = x.size(0)
    num_points = x.size(3)

    ### make 3d imu
    # x_acc = x[:,:,:3,:]
    # x_ang = x[:,:,3:,:]
    # x_resize = torch.cat((x_acc, x_ang),dim=3)
    # x_resize[:,:,:,0::2] = x_acc
    # x_resize[:,:,:,1::2] = x_ang
    # x = x_resize.reshape(batch_size, 3, -1)
    # num_points = num_points * 2
    ### make 3d imu
    
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    
    # num_dims = num_dims // 3  #for pointcloud and 3d shape imu
    num_dims = num_dims // 6  #for 6D imu

    # x = x.transpose(2, 1).contiguous()
    x_acc = x[:,:3,:].transpose(2,1).contiguous() 
    x_ang = x[:,3:,:].transpose(2,1).contiguous() 
    
    # feature = x.view(batch_size*num_points, -1)[idx, :]
    feature_acc = x_acc.view(batch_size*num_points, -1)[idx, :]
    feature_ang = x_ang.view(batch_size*num_points, -1)[idx, :]
    
    # feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    feature_acc = feature_acc.view(batch_size, num_points, k, num_dims, 3) 
    feature_ang = feature_ang.view(batch_size, num_points, k, num_dims, 3) 
    
    # x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    x_acc = x_acc.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    x_ang = x_ang.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    # print(feature.shape, x.shape)
    # cross = torch.cross(feature, x, dim=-1)
    cross_acc = torch.cross(feature_acc, x_acc, dim=-1)
    cross_ang = torch.cross(feature_ang, x_ang, dim=-1)
    
    # feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    feature_acc = torch.cat((feature_acc-x_acc, x_acc, cross_acc), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    feature_ang = torch.cat((feature_ang-x_ang, x_ang, cross_ang), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    feature = torch.cat((feature_acc,feature_ang), dim=1).contiguous()   
    # feature = torch.cat((feature,feature), dim=2).contiguous()
    # print('feature shape of getcross : ', feature.shape)     # [1024, 6, 3, 200, 20]
    return feature


def get_vector_feature(x):
    # print(x.shape)     #torch.Size([1024, 1, 6, 200])
    batch_size = x.size(0)
    num_points = x.size(3)

    ### make 3d imu
    # x_acc = x[:,:,:3,:]
    # x_ang = x[:,:,3:,:]
    # x_resize = torch.cat((x_acc, x_ang),dim=3)
    # x_resize[:,:,:,0::2] = x_acc
    # x_resize[:,:,:,1::2] = x_ang
    # x = x_resize.reshape(batch_size, 3, -1)
    # num_points = num_points * 2
    ### make 3d imu
    
    x = x.view(batch_size, -1, num_points)
    
    _, num_dims, _ = x.size()
    
    # div = 6 #for 6D imu
    div = 9 #for 6D imu + vel_body
    assert num_dims % div == 0, f"{num_dims} is not divisible by {div}"
    num_dims = int(num_dims / div)
    
    x_acc = x[:,:3,:].transpose(2,1).contiguous() 
    x_ang = x[:,3:6,:].transpose(2,1).contiguous() 
    if div == 9:
        x_vel_body = x[:,6:9,:].transpose(2,1).contiguous() 
    
    # feature = x.view(batch_size*num_points, -1)[idx, :]
    x_acc = x_acc.view(batch_size, num_points, num_dims, 3)
    x_ang = x_ang.view(batch_size, num_points, num_dims, 3)
    if div == 9:
        x_vel_body = x_vel_body.view(batch_size, num_points, num_dims, 3)
    
    if div == 6 :
        feature = torch.cat((x_acc, x_ang), dim=2).contiguous()   
    elif div == 9:
        feature = torch.cat((x_acc, x_ang, x_vel_body), dim=2).contiguous()   
    else:
        assert div == 6 or div == 9
    return feature