"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch.nn as nn
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross, get_vector_feature
import time
import torch

def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 1D convolution with kernel size 3 """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """ 1D convolution with kernel size 1 """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def vn_conv1x1(in_planes, out_planes, stride=1):
    """ 1D vn_convolution with kernel size 1 """
    if stride == 1:
        conv1x1 = nn.Linear(in_planes, out_planes, bias=False)
    elif stride == 2:
        conv1x1 = nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=False),
            VNMeanPool(stride),
        )
    # return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return conv1x1


class VN_BasicBlock1D(nn.Module):
    """ Supports: groups=1, dilation=1 """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(VN_BasicBlock1D, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv3x1(in_planes, planes, stride)
        # if stride == 1:
        #     self.conv1 = nn.Linear(in_planes, planes, bias=False)
        # elif stride == 2:
        #     self.conv1 = nn.Linear(in_planes, planes, bias=False)
        #     # self.conv1_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        #     self.conv1_pool = local_mean_pool
        # else:
        #     assert False
            
        # self.bn1 = nn.BatchNorm1d(planes)
        self.bn1 = VNBatchNorm(planes, dim=3)
                                             
        # self.relu = nn.ReLU(inplace=True)
        self.relu = VNLeakyReLU(planes,negative_slope=0.0)
        
        # print("info of conv : ", planes, planes * self.expansion, stride)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        # self.conv2 = nn.Linear(planes, planes * self.expansion, bias=False)
        
        # self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.bn2 = VNBatchNorm(planes * self.expansion, dim=3)
        
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        # x = x.unsqueeze(1) #[1024, 64, 50]
        identity = x

        # if self.stride == 1:
        #     x = torch.permute(x,(0,2,1,3))
        #     x = x.reshape(-1, x.size(2), x.size(3))
        #     out = self.conv1(x)
            
        #     # out = self.conv1(torch.transpose(x,1,-1))
        #     # out = out.transpose(1,-1)
        # elif self.stride == 2:
        #     # out = self.conv1(x.transpose(1,-1))
        #     out = self.conv1(torch.transpose(x,1,-1))
        #     out = out.transpose(1,-1)
        #     out = self.conv1_pool(out)
            
        # else:
        #     assert False
        
        x = torch.permute(x,(0,2,1,3))
        x = x.reshape(-1, x.size(2), x.size(3))
        out = self.conv1(x)
        out = out.reshape(-1, 3, out.size(1), out.size(2))
        out = torch.permute(out,(0,2,1,3))

        out = self.bn1(out)
        out = self.relu(out)

        out = torch.permute(out,(0,2,1,3))
        out = out.reshape(-1, out.size(2), out.size(3))
        out = self.conv2(out)
        out = out.reshape(-1, 3, out.size(1), out.size(2))
        out = torch.permute(out,(0,2,1,3))
        
        # print('shape of input x, size check for downsample: ', x.shape)          #[1024, 10, 6, 50]
        # print('shape of x after conv2, size check for downsample: ', out.shape)  #[1024, 21, 6, 25]
        
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            identity = identity.reshape(-1, 3, identity.size(1), identity.size(2))
            identity = torch.permute(identity,(0,2,1,3))

            identity = self.downsample[1](identity)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x1(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FcBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim):
        super(FcBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prep_channel = 128
        self.fc_dim = 512
        self.in_dim = in_dim

        # prep layer2
        self.prep1 = nn.Conv1d(
            self.in_channel, self.prep_channel, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.prep_channel)
        # fc layers
        # self.fc1 = nn.Linear(self.prep_channel * self.in_dim, self.fc_dim)
        # self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        # self.fc3 = nn.Linear(self.fc_dim, self.out_channel)
        self.fc = nn.Linear(self.prep_channel * self.in_dim, self.out_channel)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.prep1(x)
        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class VN_ResNet1D(nn.Module):
    """
    ResNet 1D
    in_dim: input channel (for IMU data, in_dim=6)
    out_dim: output dimension (3)
    len(group_sizes) = 4
    """

    def __init__(
        self,
        block_type,
        in_dim,
        out_dim,
        group_sizes,
        inter_dim,
        zero_init_residual=False,
    ):
        super(VN_ResNet1D, self).__init__()
        div = 3 #3 if vector, 1 if scalar
        self.base_plane = 64 //div
        self.inplanes = self.base_plane
        self.n_knn = 20
        
        # Input module
        
        #before vn
        # self.input_block = nn.Sequential(
        #     nn.Conv1d(
        #         in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False
        #     ),
        #     nn.BatchNorm1d(self.base_plane),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        # )
        # self.input_block_conv = nn.Conv1d(in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False)
        # self.input_block_bn = nn.BatchNorm1d(self.base_plane)
        # self.input_block_relu = nn.ReLU(inplace=True)
        # self.input_block_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        #after vn ()
        # 어떻게 feature size 변화되는지 확인하기 (encoding feature dimension check!)

        self.input_block_conv  = nn.Conv1d(in_dim//div, 64//div, kernel_size=7, stride=2, padding=3, bias=False)
        # self.input_block_conv  = nn.Linear(in_dim, self.base_plane, bias=False)
        self.input_block_local_info  = nn.Conv1d(in_dim, in_dim, kernel_size=7, stride=2, padding=3, bias=False)
        # print('num of param of conv-input-block : ' , sum(p.numel() for p in self.input_block_conv.parameters()) )
        # print('num of param of linear-input-block : ' , sum(p.numel() for p in (nn.Linear(in_dim, self.base_plane, bias=False)).parameters() ) )
        
        # self.input_block_conv  = nn.Linear(in_dim, self.base_plane, bias=False)
        self.pool = mean_pool
        self.local_pool = local_mean_pool
        self.input_block_conv_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=3)
        
        self.input_block_bn = VNBatchNorm(self.base_plane, dim=4)
        self.input_block_relu = VNLeakyReLU(self.base_plane,negative_slope=0.0)
        self.input_block_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        
        
        # Residual groups
        # self.residual_groups = nn.Sequential(
        #     self._make_residual_group1d(block_type, 64//6, group_sizes[0], stride=1),  
        #     self._make_residual_group1d(block_type, 128//6, group_sizes[1], stride=2),
        #     self._make_residual_group1d(block_type, 256//6, group_sizes[2], stride=2),
        #     self._make_residual_group1d(block_type, 512//6, group_sizes[3], stride=2),
        # )

        self.residual_groups1 = self._make_residual_group1d(block_type, 64//3, group_sizes[0], stride=1)
        # self.residual_groups2 = self._make_residual_group1d(block_type, 128//3, group_sizes[1], stride=2)
        # self.residual_groups3 = self._make_residual_group1d(block_type, 256//3, group_sizes[2], stride=2)
        self.residual_groups4 = self._make_residual_group1d(block_type, 512//3, group_sizes[3], stride=2)
        
        
        # Output module
        self.output_block1 = FcBlock(512//3*3 * block_type.expansion, out_dim, inter_dim)
        self.output_block2 = FcBlock(512//3*3 * block_type.expansion, out_dim, inter_dim)

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        # print(group_sizes[0],group_sizes[1],group_sizes[2],group_sizes[3])
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                VNBatchNorm(planes * block.expansion, dim=4), 
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))
        # print(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # print()
        # x = self.input_block(x)
        # torch.cuda.synchronize()
        # t1 = time.perf_counter()

        x = x.unsqueeze(1)
        x = get_vector_feature(x)
        x = torch.permute(x,(0,3,2,1))
        x = x.reshape(-1,x.size(2),x.size(3))
        # print('x shape after reshape : ', x.shape)   #[1024*3, 2, 200]
        
        
        # torch.cuda.synchronize()
        # t3 = time.perf_counter()
        # print('time elapsed get_graph_feature_cross: ', abs(t3-t2))
        
        # x = self.input_block_conv(torch.transpose(x,1,-1))
        x = self.input_block_conv(x)
        x = x.reshape(-1, 3, x.size(1), x.size(2))
        x = torch.permute(x,(0,2,1,3))
        
        # torch.cuda.synchronize()
        # t4 = time.perf_counter()
        # print('time elapsed input_block_conv: ', abs(t3-t4))
        
        # self.input_block_bn.to('cuda')
        x = self.input_block_bn(x)
        
        # self.input_block_relu.to('cuda')
        x = self.input_block_relu(x)
        
        # torch.cuda.synchronize()
        # t5 = time.perf_counter()
        # print('time elapsed input_block_relu: ', abs(t5-t4))
          
        # torch.cuda.synchronize()
        # t6 = time.perf_counter()
        # print('time elapsed input_block_bn: ', abs(t5-t6))
        x = self.local_pool(x,2)      
        
        # torch.cuda.synchronize()
        # t7 = time.perf_counter()
        
        x = self.residual_groups1(x)
        
        # torch.cuda.synchronize()
        # t8 = time.perf_counter()
        # print('time elapsed residual_groups1: ', abs(t8-t7))

        
        # print('shape of x after residual_groups1 : ', x.shape)  #[1024, 64, 50]   -> [1024, 10, 6, 50]  -> [1024, 21, 3, 50]
        # x = self.residual_groups2(x)
        
        # torch.cuda.synchronize()
        # t9 = time.perf_counter()
        # print('time elapsed residual_groups2: ', t9-t8)

        
        # print('shape of x after residual_groups2 : ', x.shape)  #[1024, 128, 25]  -> [1024, 21, 6, 25]
        # x = self.residual_groups3(x)
        x = self.local_pool(x,2)      
        
        # torch.cuda.synchronize()
        # t10 = time.perf_counter()
        # print('time elapsed residual_groups3: ', t10-t9)
        
        
        # print('shape of x after residual_groups3 : ', x.shape)  #[1024, 256, 13]  -> [1024, 42, 6, 13]
        x = self.residual_groups4(x)
        x = self.local_pool(x,2)      
        # print('shape of x after residual_groups4 : ', x.shape)  #[1024, 512, 7]  -> [1024, 170, 3, 7]
        
        # >>> SO(3) Equivariance Check
        # print('value x after encoder : ',x[:1,:1,:,0])    
        # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # x = torch.matmul(rotation_matrix, x.permute(2,0,1,3).reshape(3,-1)).reshape(3,x.size(0),x.size(1),x.size(3)).permute(1,2,0,3)
        # print('rotated value x after encoder : ', x[:1,:1,:,0])x    
        # <<< SO(3) Equivariance Check
           
        x = x.reshape(x.shape[0], -1, x.shape[-1])   #[1024, 510, 7]
        
        # torch.cuda.synchronize()
        # t11 = time.perf_counter()
        # print('time elapsed residual_groups4: ', t11-t10)
                 
        mean = self.output_block1(x)  # mean
        logstd = self.output_block2(x)  # covariance sigma = exp(2 * logstd)
        # params_outblock = sum(p.numel() for p in self.output_block1.parameters()) + sum(p.numel() for p in self.output_block2.parameters())
        # print('shape of x after output_block : ', mean.shape, logstd.shape, 'num of params in outblock : ', params_outblock)  #[1024, 512, 7]
        
        # torch.cuda.synchronize()
        # t12 = time.perf_counter()
        # print('time elapsed output_block 1 & 2: ', t12-t11)
        # print()
        
        # print(self.output_block1)
        # print(self.output_block2)
        # print()
        
        return mean, logstd
