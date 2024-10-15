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


class VN_BasicBlock1D_cl(nn.Module):
    """ Supports: groups=1, dilation=1 """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(VN_BasicBlock1D_cl, self).__init__()
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
        # print('x shape before prep1 : ', x.shape, self.prep1.weight.shape)  #[1024, 170, 3, 7] -> [1024, 510, 7] -> [1024*3, 170, 7]
        x = self.prep1(x)
        
        # print('x shape after prep1 : ', x.shape)  #[1024, 170, 3, 7] -> [1024, 510, 7] -> [1024*3, 170, 7]
        x = x.reshape(-1, 3, x.size(1), x.size(2))
        x = torch.permute(x,(0,2,1,3))
        
        # x = self.bn1.to('cuda')(x)
        x = self.bn1(x)
        # print('x shape after bn1 : ', x.shape)  #[1024, 42, 3, 7] -> ln : [1024, 42, 3, 32]
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


class VN_ResNet1D_cl(nn.Module):
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
        super(VN_ResNet1D_cl, self).__init__()
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
        print(in_dim//div, 64//div)
        self.input_block_conv  = nn.Conv1d(in_dim//div, 64//div, kernel_size=7, stride=2, padding=3, bias=False)
        # self.input_block_conv  = nn.Linear(in_dim, self.base_plane, bias=False)
        self.input_block_local_info  = nn.Conv1d(in_dim, in_dim, kernel_size=7, stride=2, padding=3, bias=False)
        # print('num of param of conv-input-block : ' , sum(p.numel() for p in self.input_block_conv.parameters()) )
        # print('num of param of linear-input-block : ' , sum(p.numel() for p in (nn.Linear(in_dim, self.base_plane, bias=False)).parameters() ) )
        # self.input_block_conv  = nn.Linear(in_dim, self.base_plane, bias=False)
        self.pool = mean_pool
        # self.local_pool = local_mean_pool
        self.local_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
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
        self.residual_groups2 = self._make_residual_group1d(block_type, 128//3, group_sizes[1], stride=2)
        self.residual_groups3 = self._make_residual_group1d(block_type, 256//3, group_sizes[2], stride=2)
        self.residual_groups4 = self._make_residual_group1d(block_type, 512//3, group_sizes[3], stride=2)
        
        
        # Output module
        self.output_block1 = FcBlock(512//3*3 * block_type.expansion, out_dim, inter_dim)
        # diagonal : 1, pearson : 2, direct covariance : 3
        covariance_param = 1
        self.output_block2 = FcBlock(512//3*3 * block_type.expansion, out_dim*covariance_param, inter_dim)
        # self.output_block3 = FcBlock(512//3*3 * block_type.expansion, out_dim, inter_dim)

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
                # if isinstance(m, Bottleneck1D):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, VN_BasicBlock1D_cl):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, previous_output):
        # x = self.input_block(x)
        previous_output = previous_output.unsqueeze(-1).repeat(1,1,200).to("cuda")
        x = torch.cat((x, previous_output), dim=1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        x = x.unsqueeze(1)
        x = get_vector_feature(x)
        
        x = torch.permute(x,(0,3,2,1))
        x = torch.reshape(x,(-1,x.size(2),x.size(3)))
        
        x = self.input_block_conv(x)
        x = torch.reshape(x,(-1, 3, x.size(1), x.size(2)))
        x = torch.permute(x,(0,2,1,3))
        
        x = self.input_block_bn(x)
        x = self.input_block_relu(x)

        b,c,h,w = x.shape
        x = self.local_pool(x.reshape(-1, h,w))  
        x = torch.reshape(x,(b,c,x.shape[1],x.shape[2]))
        
        x = self.residual_groups1(x)
        x = self.residual_groups2(x)
        x = self.residual_groups3(x)
        x = self.residual_groups4(x)

        mean = self.output_block1(x)  # mean
        covariance = self.output_block2(x)  # pearson : [n x 6] or diagonal : [n x 3]
        
        return mean, covariance

class SO3EquivariantReluBracketLayers(nn.Module):
    def __init__(self, in_dim, out_dim, inter_dim):
        super(SO3EquivariantReluBracketLayers, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        div = 3
        self.num_m_feature = 64
        self.vn_bn = VNBatchNorm(64, dim=3)
        self.ln_bracket = LNLinearAndLieBracket(200, 64, share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.vnmaxpool = VNMaxPool(64)
        self.R = torch.eye(3).to("cuda")
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.m2 = nn.Parameter(torch.zeros((3,64)))
        self.m3 = nn.Parameter(torch.zeros((3,128)))
        self.m4 = nn.Parameter(torch.zeros((3,256)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m1 = LNLinearAndvnrelu(self.num_m_feature,3,algebra_type='so3') <<< choose 너가 원하는 relu 골라!
        
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(64,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(64,3,algebra_type='so3')
        self.map_m_to_m5 = LNLinearAndKillingRelu(128,3,algebra_type='so3')
        self.map_m_to_m6 = LNLinearAndKillingRelu(128,3,algebra_type='so3')
        self.map_m_to_m7 = LNLinearAndKillingRelu(256,3,algebra_type='so3')
        self.map_m_to_m8 = LNLinearAndKillingRelu(256,3,algebra_type='so3')
        self.ln_channel_mix_residual1 = LNLinearAndLieBracketChannelMix(self.num_m_feature,64,algebra_type='so3',residual_connect=False)
        self.ln_channel_mix_residual2 = LNLinearAndLieBracketChannelMix(64,128,algebra_type='so3',residual_connect=False)
        self.ln_channel_mix_residual3 = LNLinearAndLieBracketChannelMix(128,256,algebra_type='so3',residual_connect=False)
        self.ln_channel_mix_residual4 = LNLinearAndLieBracketChannelMix(256,512,algebra_type='so3',residual_connect=False)
        
        self.output_block1 = FcBlock(32*3, out_dim, 32)
        self.output_block2 = FcBlock(32*3, out_dim, 32)
        
        self.ln_fc = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc2 = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc_bracket2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

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
        # print(x.shape)
        x = rearrange(x,'b c d f -> (b d) f c')
        
        # print(x.shape)
        x = self.ln_bracket(x)
        # print(x.shape)  #[2048, 64, 3]
        # b,c,h = x.shape
        # x = self.vnmaxpool(x)   #[2048, 64] 
        # print(x.shape)
        # x = torch.reshape(x,(b,c,x.shape[1]))
        
        # x = torch.reshape(x,(-1, 3, x.size(1), x.size(2)))
        # x = torch.permute(x,(0,2,1,3))
        x = self.vn_bn(x)
        
        # print(x.shape)  #[2048, 64, 3]
        
        x_reshape = rearrange(x,'b f d -> b f d 1')
        m1 = self.map_m_to_m1(rearrange(x),'k f -> 1 f k 1')
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m2),'k f -> 1 f k 1'))
        m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m2),'k f -> 1 f k 1'))
        m5 = self.map_m_to_m5(rearrange(torch.matmul(self.R,self.m3),'k f -> 1 f k 1'))
        m6 = self.map_m_to_m6(rearrange(torch.matmul(self.R,self.m3),'k f -> 1 f k 1'))
        m7 = self.map_m_to_m7(rearrange(torch.matmul(self.R,self.m4),'k f -> 1 f k 1'))
        m8 = self.map_m_to_m8(rearrange(torch.matmul(self.R,self.m4),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        m3 = rearrange(m3,'1 f k 1 -> k f')
        m4 = rearrange(m4,'1 f k 1 -> k f')
        m5 = rearrange(m5,'1 f k 1 -> k f')
        m6 = rearrange(m6,'1 f k 1 -> k f')
        m7 = rearrange(m7,'1 f k 1 -> k f')
        m8 = rearrange(m8,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        M3 = torch.matmul(m3,m3.transpose(0,1))
        M4 = torch.matmul(m4,m4.transpose(0,1))
        M5 = torch.matmul(m5,m5.transpose(0,1))
        M6 = torch.matmul(m6,m6.transpose(0,1))
        M7 = torch.matmul(m7,m7.transpose(0,1))
        M8 = torch.matmul(m8,m8.transpose(0,1))
        
        # print(x.shape)
        x = self.ln_channel_mix_residual1(x_reshape, M1, M2)
        # print(x.squeeze().shape)
        x = self.ln_channel_mix_residual2(x, M3, M4)
        # print(x.squeeze().shape)
        x = self.ln_channel_mix_residual3(x, M5, M6)
        # print(x.squeeze().shape)
        x = self.ln_channel_mix_residual4(x, M7, M8)
        # print(x.squeeze().shape)    #[2048, 512, 3]
        x = rearrange(x,'b f d 1 -> b f d')
        x = rearrange(x, '(b1 b2) h c -> b1 c (b2 h)', b2=2)
        x = rearrange(x, 'b1 c (h w) -> b1 h c w', h=32, w=32)

        mean = self.output_block1(x)  
        covariance = self.output_block2(x) 
        

        # x = self.ln_fc_bracket(x)  # [B, F, 3, 1]
        # x = self.ln_fc(x)   # [B, F, 3, 1]
        # x = self.ln_fc_bracket2(x)
        # x = self.ln_fc2(x)

        # x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
        # x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 3]
        # # x_out = rearrange(self.ln_pooling(x), 'b 1 k 1 -> b k')   # [B, F, 1, 1]
        
        
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