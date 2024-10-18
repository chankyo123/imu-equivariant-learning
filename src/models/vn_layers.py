import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # print('shape of x in the input of vnleakyrelu : ', x.shape)
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        # print('shape of x at the output of vnleakyrelu : ', x_out.shape)
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        print('shape of x before x.transpose(1,-1) : ', x.shape)
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        print('shape of x after x.transpose(1,-1) : ', x.transpose(1,-1).shape)
        print('shape of x after nn.Linear : ', p.shape)
        # BatchNorm
        p = self.batchnorm(p)
        print('shape of x after VNBatchNorm : ', p.shape)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        print('shape of x after VN LeakyReLU : ', x_out.shape)
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            # self.bn = nn.BatchNorm1d(num_features).to('cuda')
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            # self.bn = nn.BatchNorm2d(num_features).to('cuda')
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # print("input shape of vnmaxpool : ", x.shape)
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        # print("output shape of vnmaxpool : ", x_max.shape)
        return x_max
    
class VNMeanPool(nn.Module):
    def __init__(self, kernel=2):
        super(VNMeanPool, self).__init__()
        self.kernel = kernel    
    def forward(self, x):
        B,C,N,W = x.shape
        if isinstance(W, torch.Tensor):
            W = W.item()
        reduced_W = (W + 1)//2 if self.kernel == 2 and W % 2 == 1 else W //self.kernel
        output = torch.zeros(B,C,N,reduced_W).to('cuda')

        for i in range(reduced_W):
            local_region = x[:, :, :, i * self.kernel : (i+1) * self.kernel]
            local_average = local_region.mean(dim=3, keepdim=False)
            output[:,:,:,i] = local_average
        # print('input shape of vnmeanpool : ', x.shape)
        # print('output shape of vnmeanpool : ', output.shape)
        return output
            
    


def mean_pool(x, dim=-1, keepdim=False):
    # print("input shape of mean_pool : ", x.shape)
    # print("output shape of mean_pool : ", (x.mean(dim=dim, keepdim=keepdim)).shape)
    return x.mean(dim=dim, keepdim=keepdim)


class VNMeanPool_local(nn.Module):
    def __init__(self, kernel=2):
        super(VNMeanPool_local, self).__init__()
        self.kernel = kernel    
    def forward(self, x):
        if len(x.shape) == 4:
            B,C,N,W = x.shape
            if isinstance(W, torch.Tensor):
                W = W.item()
            reduced_W = (W + 1)//2 if self.kernel == 2 and W % 2 == 1 else W //self.kernel
            output = torch.zeros(B,C,N,reduced_W).to('cuda')

            for i in range(reduced_W):
                local_region = x[:, :, :, i * self.kernel : (i+1) * self.kernel]
                local_average = local_region.mean(dim=3, keepdim=False)
                output[:,:,:,i] = local_average
            return output
        else:
            B,N,W = x.shape
            if isinstance(W, torch.Tensor):
                W = W.item()
            reduced_W = (W + 1)//2 if self.kernel == 2 and W % 2 == 1 else W //self.kernel
            output = torch.zeros(B,N,reduced_W).to('cuda')
            
            for i in range(reduced_W):
                local_region = x[:, :, i * self.kernel : (i+1) * self.kernel]
                local_average = local_region.mean(dim=2, keepdim=False)
                output[:,:,i] = local_average
            return output
            
            
def local_mean_pool(x, kernel=2):
    if len(x.shape) == 4:
        B,C,N,W = x.shape
        if isinstance(W, torch.Tensor):
            W = W.item()
            
        if kernel == 2 and W % 2 == 1:
            reduced_W = (W + 1)//2
        else:
            reduced_W = W // kernel
            
        output = torch.zeros(B,C,N,reduced_W).to('cuda')
        
        for i in range(reduced_W):
            local_region = x[:,:,:,i*kernel:(i+1)*kernel]
            local_average = local_region.mean(dim=3, keepdim=False)
            output[:,:,:,i] = local_average
        
        return output.to('cuda')
    else:
        B,N,W = x.shape
        if isinstance(W, torch.Tensor):
            W = W.item()
            
        if kernel == 2 and W % 2 == 1:
            reduced_W = (W + 1)//2
        else:
            reduced_W = W // kernel
            
        output = torch.zeros(B,N,reduced_W).to('cuda')
        
        for i in range(reduced_W):
            local_region = x[:,:,i*kernel:(i+1)*kernel]
            local_average = local_region.mean(dim=2, keepdim=False)
            output[:,:,i] = local_average
        
        return output.to('cuda')


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0