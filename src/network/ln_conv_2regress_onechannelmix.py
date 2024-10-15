"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch.nn as nn
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross, get_vector_feature
from models.lie_alg_util import *
from models.lie_neurons_layers import *
import time
import torch
from fvcore.nn import FlopCountAnalysis

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


class LN_BasicBlock1D(nn.Module):
    """ Supports: groups=1, dilation=1 """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(LN_BasicBlock1D, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv3x1(in_planes, planes, stride)
        # if stride == 1:
        #     self.conv1 = nn.Linear(in_planes, planes, bias=False)
        # elif stride == 2:
        #     self.conv1 = nn.Linear(in_planes, planes, bias=False)
        #     self.conv1_pool = VNMeanPool_local(2)
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
        identity = x.clone()

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
        self.in_channel = in_channel//3
        self.out_channel = out_channel//3
        self.prep_channel = 128//3
        self.fc_dim = 512//3
        self.in_dim = in_dim

        # prep layer2
        self.prep1 = nn.Conv1d(
            self.in_channel, self.prep_channel, kernel_size=1, bias=False
        )
        self.bn1 = VNBatchNorm(self.prep_channel, dim=3)
        # fc layers
        ## TODO: check if linear contain bias
        self.fc1 = nn.Linear(self.prep_channel * self.in_dim, self.fc_dim, bias = False)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim, bias = False)
        self.fc3 = nn.Linear(self.fc_dim, self.out_channel, bias = False)
        # self.fc1 = nn.Linear(self.prep_channel * self.in_dim, self.fc_dim)
        # self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        # self.fc3 = nn.Linear(self.fc_dim, self.out_channel)
        self.relu = VNLeakyReLU(self.fc_dim,negative_slope=0.0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.prep1.to('cuda')(x)
        # print('x shape after residual block:  ', x.shape)  #[1024, 510, 7] -> [1024, 170, 3, 7]
        x = torch.permute(x,(0,2,1,3)) 
        x = x.reshape(-1,x.size(2),x.size(3))
        # print('x shape before prep1 : ', x.shape)  #[1024, 170, 3, 7] -> [1024, 510, 7] -> [1024*3, 170, 7]
        x = self.prep1(x)
        
        # print('x shape after prep1 : ', x.shape)  #[1024, 170, 3, 7] -> [1024, 510, 7] -> [1024*3, 170, 7]
        x = x.reshape(-1, 3, x.size(1), x.size(2))
        x = torch.permute(x,(0,2,1,3))
        
        # x = self.bn1.to('cuda')(x)
        x = self.bn1(x)
        # print('x shape after bn1 : ', x.shape)  #[1024, 42, 3, 7]
        x = torch.permute(x,(0,2,1,3)) 
        x = x.reshape(x.size(0),x.size(1), -1)
        # x = self.fc1.to('cuda')(x)
        # print('self.fc1 weight : ', self.fc1.weight.shape) #[512, 896]
        x = self.fc1(x)
        # print('x shape after fc1 : ', x.shape)  #[1024, 512] -> [1024, 3, 170]
        x = torch.permute(x,(0,2,1)) 
        x = self.relu(x)
        
        # print('x shape after relu : ', x.shape)  #[1024, 170, 3]
        # x = self.dropout(x)
        
        # x = self.fc2.to('cuda')(x)
        x = torch.permute(x,(0,2,1)) 
        x = self.fc2(x)
        # print('x shape after fc2 : ', x.shape)  #[1024, 3, 170]
        x = torch.permute(x,(0,2,1)) 
        x = self.relu(x)
        
        # x = self.dropout(x)   
        
        # x = self.fc3.to('cuda')(x)
        x = torch.permute(x,(0,2,1)) 
        x = self.fc3(x)
        x = torch.permute(x,(0,2,1)) 
        
        # if self.fc1.bias is not None:
        #     print("The fc1 layer has a bias term.")
        # else:
        #     print("The fc1 layer does not has a bias term.")
            
        # if self.fc2.bias is not None:
        #     print("The fc2 layer has a bias term.")
        # else:
        #     print("The fc2 layer does not has a bias term.")
        # if self.fc3.bias is not None:
        #     print("The fc3 layer has a bias term.")
        # else:
        #     print("The fc3 layer does not has a bias term.")
            
            
        if self.out_channel == 1: #[n x 3]
            x = x.squeeze(dim=1)

        elif self.out_channel == 2: #[n x 6]
            x = x.reshape(x.size(0), -1)
        
        # # >>> SO(3) Equivariance Check
        # if self.out_channel == 1: 
        #     print('value x after layers : ',x[:1,:])    
        #     rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        #     rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        #     x = torch.matmul(rotation_matrix, x.permute(1,0).reshape(3,-1)).reshape(3,x.size(0)).permute(1,0)
        #     print('rotated value x after layers : ', x[:1,]) 
        # # <<< SO(3) Equivariance Check
        
        # print('x shape after fc3 : ', x.shape)  #[1024, 3, 1]
        return x

class LNLinear_VNBatch_KillingRelu(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='so3', share_nonlinearity=False, leaky_relu=False,negative_slope=0.2):
        super(LNLinear_VNBatch_KillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.linear = LNLinear(in_channels, out_channels)
        self.ln_bn = VNBatchNorm(out_channels, dim=4)
        self.leaky_relu = LNKillingRelu(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, negative_slope=negative_slope)

    def forward(self, x, M1=torch.eye(3), M2=torch.eye(3)):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x = self.linear(x)
        x = self.ln_bn(x)
        x_out = self.leaky_relu(x)
        return x
    
class LNLinear_VNBatch_VNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='so3', share_nonlinearity=False, leaky_relu=False,negative_slope=0.2):
        super(LNLinear_VNBatch_VNRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.linear = LNLinear(in_channels, out_channels)
        self.ln_bn = VNBatchNorm(out_channels, dim=4)
        self.leaky_relu = VNLeakyReLU(
            out_channels,
            # algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu,
            negative_slope=negative_slope)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x = self.linear(x)
        x = self.ln_bn(x)
        x_out = self.leaky_relu(x)
        return x
    
class LNLinear_VNBatch_Liebracket(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='so3', share_nonlinearity=False, negative_slope=0.2):
        super(LNLinear_VNBatch_Liebracket, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.linear = LNLinear(in_channels, out_channels)
        self.ln_bn = VNBatchNorm(out_channels, dim=4)
        self.leaky_relu = LNLieBracket(out_channels, algebra_type='so3', share_nonlinearity=share_nonlinearity)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x = self.linear(x)
        x = self.ln_bn(x)
        x_out = self.leaky_relu(x)
        return x
    
class SO3EquivariantReluBracketLayers(nn.Module):
    def __init__(self, 
                block_type,
                in_dim, out_dim, 
                group_sizes,
                inter_dim, zero_init_residual=True):
        super(SO3EquivariantReluBracketLayers, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        div = 3
        
        self.num_m_feature = 64
        self.first_out_channel = 64
        self.inplanes = self.first_out_channel//div
        self.vn_bn = VNBatchNorm(64//3, dim=3)
        # self.ln_bracket = LNLinearAndLieBracket(in_dim//3, self.num_m_feature//3, share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.ln_linear = LNLinear(in_dim//3, self.first_out_channel//3)
        self.ln_conv = nn.Conv1d(in_dim//div, self.first_out_channel//div, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_block_bn = VNBatchNorm(self.first_out_channel//3, dim=4)
        # self.input_block_relu = VNLeakyReLU(self.first_out_channel//3,negative_slope=0.0)
        self.liebracket = LNLieBracket(self.first_out_channel//3, algebra_type='so3', share_nonlinearity=share_nonlinearity)
        
        self.ln_pool = VNMeanPool_local(2)
        
        # self.map_m_to_m1 = LNLinear_VNBatch_VNRelu(self.first_out_channel//div,self.first_out_channel//div, negative_slope=0.0)
        self.map_m_to_m1 = LNLinear_VNBatch_Liebracket(self.first_out_channel//div,self.first_out_channel//div, negative_slope=0.0)
        # self.map_m_to_m2 = LNLinear_VNBatch_Liebracket(256//div,256//div)

        self.output_block1 = FcBlock(512//3*3, out_dim, 7)
        self.output_block2 = FcBlock(512//3*3, out_dim, 7)
        
        self.residual_groups1 = self._make_residual_group1d(block_type, self.first_out_channel//3, group_sizes[0], stride=1)
        # self.residual_groups2 = self._make_residual_group1d(block_type, 128//3, group_sizes[1], stride=2)
        # self.residual_groups3 = self._make_residual_group1d(block_type, 256//3, group_sizes[2], stride=2)
        self.residual_groups4 = self._make_residual_group1d(block_type, 512//3, group_sizes[3], stride=2)
        # self.ln_conv2 = nn.Conv1d(512//3, 512//3, kernel_size=7, stride=2, padding=3, bias=False)
        
        # self.ln_fc = LNLinearAndKillingRelu(
        #     feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        # self.ln_fc2 = LNLinearAndKillingRelu(
        #     feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        # self.ln_fc_bracket2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        # self.fc_final = nn.Linear(feat_dim, 1, bias=False)
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
            # print(type(m))
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
                # if isinstance(m, Bottleneck1D):
                #     nn.init.constant_(m.bn3.weight, 0)
                # print(m)
                # if isinstance(m, LNLinearAndLieBracketChannelMix_VNBatchNorm):
                #     nn.init.constant_(m.ln_bn.bn.weight, 0)
                if isinstance(m, LN_BasicBlock1D):
                    nn.init.constant_(m.bn2.bn.weight, 0)
                    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        
        # x = self.input_block_conv(x)
        x = x.unsqueeze(1)
        x = get_vector_feature(x)
        x = torch.permute(x,(0,3,2,1))
        
        # x = torch.reshape(x,(-1,x.size(2),x.size(3)))
        # print(x.shape) #[1024, 3, 2, 200]
        
        
        # print(x.shape)
        # x = rearrange(x,'b c d f -> b d c f')
        # x = self.ln_bracket(x)
        # print(self.ln_bracket)
        # x = self.ln_pool(x)
        # x = self.ln_pool(x)
        
        # linear_flops = FlopCountAnalysis(self.ln_linear, x_linear)
        # print(f"linear-FLOPs: {linear_flops.total()}")   # 25804800

        # x = self.ln_linear(x)
        # x = self.ln_pool(x)
        # x = self.vn_bn(x)
        # print(x.shape)   #[1024, 3, 2, 200]
        # conv_flops = FlopCountAnalysis(self.ln_conv, x)
        # print(f"conv-FLOPs: {conv_flops.total()}")   #90316800
        
        x = torch.reshape(x,(-1,x.size(2),x.size(3)))
        x = self.ln_conv(x)
        x = torch.reshape(x,(-1, 3, x.size(1), x.size(2)))
        x = torch.permute(x,(0,2,1,3))
        x = self.input_block_bn(x)
        b,c,h,w = x.shape
        #1.
        # x = self.input_block_relu(x)
        #1.
        #2.
        x = self.liebracket(x)
        #2.
        x = self.ln_pool(x)
        
        # print(x.shape)  #[1024, 21, 3, 50] -> [1024, 85, 3, 50]
        x = torch.reshape(x,(b,c,h,-1))
        x = self.residual_groups1(x)
        m1 = self.map_m_to_m1(x)
        M1 = torch.einsum("b f k d, b k e d -> b f e d", m1, m1.transpose(1,2))
        x = torch.einsum("b f k d, b k e d -> b f e d", M1,x)
        
        # x = self.residual_groups2(x)
        # x = self.residual_groups3(x)
        
        # m2 = self.map_m_to_m2(x)
        # M2 = torch.einsum("b f k d, b k e d -> b f e d", m2, m2.transpose(1,2))
        # x = torch.einsum("b f k d, b k e d -> b f e d", M2,x)
        x = self.ln_pool(x)
        
        x = self.residual_groups4(x)
        
        
        x = self.ln_pool(x)


        # x = rearrange(x,'b f w d -> (b w) f d')
        # x = self.ln_conv2(x)
        # x = torch.reshape(x,(-1, 3, x.size(1), x.size(2)))
        # x = torch.permute(x,(0,2,1,3))
        
        # x = self.ln_channel_mix_residual2(x, M3, M4)
        # x = self.ln_channel_mix_residual3(x, M5, M6)
        # x = self.ln_channel_mix_residual4(x, M7, M8)
        # x = rearrange(x,'b (f w) d 1 -> b f d w', w = 7)
        # print(x.shape)  #[1024, 170, 3, 7]

        mean = self.output_block1(x)  
        covariance = self.output_block2(x) 
        
        # # >>> SO(3) Equivariance Check : (3,1) vector and (3,3) covariance
         
        # print('value x after layers : ',mean[:1,:])    
        # print()
        # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # mean_rot = torch.matmul(rotation_matrix, mean.permute(1,0)).permute(1,0)
        # print('rotated value x after layers : ', mean_rot[:1,]) 
        
        # #1. using tlio's cov
        # # print('covariance after layers : ',covariance[:1,:])    
        # # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # # covariance_rot = torch.matmul(torch.matmul(rotation_matrix, covariance.permute(1,0)).permute(1,0), rotation_matrix.T)
        # # print('rotated value x after layers : ', covariance_rot[:1,:]) 
        
        # #2. using 3*3 cov
        # # print('covariance after layers : ',covariance[:1,:,:])    
        # # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # # covariance_rot = torch.matmul(torch.matmul(rotation_matrix, covariance.permute(1,2,0)).permute(2,0,1), rotation_matrix.T)
        # # print('rotated value x after layers : ', covariance_rot[:1,:,:]) 
        # # <<< SO(3) Equivariance Check
        return mean, covariance