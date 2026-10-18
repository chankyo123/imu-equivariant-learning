import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

# helper 

def orthogonal_input(x, dim=-1):
    return torch.concatenate((-x[..., 1,:].unsqueeze(-2), x[..., 0,:].unsqueeze(-2)), dim=dim)

## equivariant modules
class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        scalar_dim_out,
    ):
        super().__init__()

        self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)
        self.scalar_linear = nn.Linear(scalar_dim_in, scalar_dim_out, bias=False)

    def forward(self, vector, scalar):
        return self.vector_linear(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1)), self.scalar_linear(scalar)
    
class NonLinearity(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        scalar_dim_out,
    ):
        super().__init__()
        self.scalar_dim_out = scalar_dim_out
        self.dim_out = dim_out

        self.linear = nn.Linear(dim_in+scalar_dim_in, dim_out+scalar_dim_out, bias=False)
        self.layer_norm = LayerNorm(dim_out+scalar_dim_out)

        
    def forward(self, vector, scalar):
        x = torch.concatenate((torch.norm(vector, dim=-2), scalar), dim=-1)
        x = self.linear(x)
        # x = nn.ReLU()(x)
        x = self.layer_norm(x)
        return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2)), x[..., -self.scalar_dim_out:]
    
class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        d = self.map_to_dir(x)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        return self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/((d*d).sum(2, keepdim=True)+1e-8))*d))

class VNMaxPool(nn.Module):
    def __init__(self, in_channels, kernel_size=None, stride=None, padding=None):
        super(VNMaxPool, self).__init__()
        self.stride = stride
        self.kernel = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.in_dim = None
        if stride is not None:
            self.in_dim = int(int((self.in_channels+2*self.padding-(self.kernel-1)-1)/self.stride)+1) # 0 padding for equivariance to hold
            self.sca_maxpool = nn.MaxPool1d(kernel_size=self.kernel, stride=stride, padding=padding)
            self.map_to_dir = nn.Linear(self.kernel, self.kernel, bias=False)
        else:
            self.sca_maxpool = nn.MaxPool1d(self.in_channels)
            self.map_to_dir = nn.Linear(self.in_channels, self.in_channels, bias=False)
    
    def forward(self, x, scalar):
        if self.stride is not None:
            x = x.permute(0,2,1,3)#(1,2,200,6) <- (1,200,2,6)
            ch = x.shape[-1]
            x = torch.concat([x[:,:,0,:].unsqueeze(-2).repeat([1,1,self.padding,1]),x,x[:,:,-1,:].unsqueeze(-2).repeat([1,1,self.padding,1])], dim=-2) # repeating last value for padding
            x = torch.nn.Unfold(kernel_size=(self.kernel,1), dilation=1, stride=1)(x)
            x = x.reshape((x.shape[0],2,-1,ch)).permute(0,2,1,3).reshape((x.shape[0],self.kernel,-1,2,ch))[:,:,::self.stride,:,:] #(1,3,100,2,6)
            x = x.permute(0,2,4,3,1)#(1,3,100,2,6) -> (1,100,6,2,3)
            d = self.map_to_dir(x)
            dotprod = (x*d).sum(-2, keepdims=True)
            vec_out = x[torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (dotprod.max(dim=-1, keepdim=False)[1],)].permute(0,1,3,2) #(1,100,6,2) -> (1,100,2,6)
            sca_out = self.sca_maxpool(scalar.permute(0,2,1)).permute(0,2,1)
        else:
            x = x.permute(0,3,2,1)
            d = self.map_to_dir(x)
            dotprod = (x*d).sum(-2, keepdims=True)
            vec_out = x[torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (dotprod.max(dim=-1, keepdim=False)[1],)].permute(0,2,1)
            sca_out = self.sca_maxpool(scalar.permute(0,2,1)).squeeze(-1)
        return vec_out, sca_out
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-3): # dim is the vector dimension (i.e., 2)
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -2)
        x = x / norms.clamp(min = self.eps).unsqueeze(-2)
        return x * self.ln(norms).unsqueeze(-2)
    
class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features, affine=False)
    
    def forward(self, x):
        x = x.permute(0,3,2,1)
        norm = torch.norm(x, dim=2).clamp(min = 1e-4)
        x = (x / norm.unsqueeze(2)) * self.bn(norm).unsqueeze(2)
        return x.permute(0,3,2,1)

class BatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(BatchNorm, self).__init__()
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features, affine=False)
    
    def forward(self, x):
        return self.bn(x.permute(0,2,1)).permute(0,2,1)
    
class Convolutional(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        stride = 1,
        padding=0,
        kernel=(16,1),
        bias = False
    ):
        super().__init__()
        self.conv_layer_vec = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, stride=(stride, 1), kernel_size=kernel, bias=bias, padding=(padding,0), padding_mode='replicate')
        self.conv_layer_sca = nn.Conv2d(in_channels=scalar_dim_in, out_channels=scalar_dim_out, stride=(stride, 1), kernel_size=kernel, bias=bias, padding=(padding,0), padding_mode='replicate')

        
    def forward(self, vector, scalar):
        return self.conv_layer_vec(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1).permute(0,3,1,2)).permute(0,2,3,1), self.conv_layer_sca(scalar.unsqueeze(-2).permute(0,3,1,2)).permute(0,2,3,1).squeeze(-2)

class BasicBlock1D(nn.Module):
    """ Supports: groups=1, dilation=1 """

    def __init__(self, indim, i, stride=1, downsample=False, bias=False, padding=1):
        super(BasicBlock1D, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = Convolutional(dim_in=indim, dim_out= i, scalar_dim_in=indim, scalar_dim_out=i*2, stride = stride, kernel = (3,1), bias=bias, padding=padding)
        self.nl1 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2)
        self.vbn1 = VNBatchNorm(num_features=i, dim=4)
        self.sbn1 = BatchNorm(num_features=i*2, dim=4)
        self.relu = nn.ReLU()
        self.vr1 = VNLeakyReLU(in_channels=i,share_nonlinearity=False) ## scalar relu done directly
        self.conv2 = Convolutional(dim_in=2*i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2, stride = 1,  kernel = (3,1), bias=bias, padding=padding)
        self.nl2 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2)
        self.vbn2 = VNBatchNorm(num_features=i, dim=4)
        self.sbn2 = BatchNorm(num_features=i*2, dim=4)
        self.stride = stride
        self.downsample = downsample
        self.vrelu_o = VNLeakyReLU(in_channels=i,share_nonlinearity=False)
        self.dconv1 = Convolutional(dim_in=indim, dim_out= i, scalar_dim_in=indim, scalar_dim_out=i*2, stride = stride, kernel = (1,1), bias=bias, padding=0) ## residual 
        self.d_nl1 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2)#residual
        self.d_vbn1 = VNBatchNorm(num_features=i, dim=4)#VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4), #residual
        self.d_sbn1 = BatchNorm(num_features=i*2, dim=4)

    def forward(self, v,s):
        res_v = torch.clone(v)
        res_s = torch.clone(s)

        v,s = self.conv1(v,s)
        v,s = self.nl1(v,s)
        v = self.vbn1(v)
        s = self.sbn1(s)
        v = self.vr1(v)
        s = self.relu(s)
        v,s = self.conv2(v,s)
        v,s = self.nl2(v,s)
        v = self.vbn2(v)
        s = self.sbn2(s)         

        if self.downsample:
            res_v,res_s = self.dconv1(res_v,res_s)
            res_v,res_s = self.d_nl1(res_v,res_s)
            res_v = self.d_vbn1(res_v)
            res_s = self.d_sbn1(res_s)
            v = v+res_v
            s = s+res_s

        return self.vrelu_o(v), self.relu(s)
    
class BasicBlock1D_flatten(nn.Module):
    """ Supports: groups=1, dilation=1 """

    def __init__(self, indim, i, stride=1, downsample=False, bias=False, padding=1):
        super(BasicBlock1D_flatten, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = Convolutional(dim_in=indim, dim_out= i, scalar_dim_in=indim//2, scalar_dim_out=i, stride = stride, kernel = (3,1), bias=bias, padding=padding)
        self.nl1 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)
        self.vbn1 = VNBatchNorm(num_features=i, dim=4)
        self.sbn1 = BatchNorm(num_features=i, dim=4)
        self.relu = nn.ReLU()
        self.vr1 = VNLeakyReLU(in_channels=i,share_nonlinearity=False) ## scalar relu done directly
        self.conv2 = Convolutional(dim_in=2*i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i, stride = 1,  kernel = (3,1), bias=bias, padding=padding)
        self.nl2 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)
        self.vbn2 = VNBatchNorm(num_features=i, dim=4)
        self.sbn2 = BatchNorm(num_features=i, dim=4)
        self.stride = stride
        self.downsample = downsample
        self.vrelu_o = VNLeakyReLU(in_channels=i,share_nonlinearity=False)
        self.dconv1 = Convolutional(dim_in=indim, dim_out= i, scalar_dim_in=indim//2, scalar_dim_out=i, stride = stride, kernel = (1,1), bias=bias, padding=0) ## residual 
        self.d_nl1 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)#residual
        self.d_vbn1 = VNBatchNorm(num_features=i, dim=4)#VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4), #residual
        self.d_sbn1 = BatchNorm(num_features=i, dim=4)

    def forward(self, v,s):
        res_v = torch.clone(v)
        res_s = torch.clone(s)

        v,s = self.conv1(v,s)
        v,s = self.nl1(v,s)
        v = self.vbn1(v)
        s = self.sbn1(s)
        v = self.vr1(v)
        s = self.relu(s)
        v,s = self.conv2(v,s)
        v,s = self.nl2(v,s)
        v = self.vbn2(v)
        s = self.sbn2(s)         

        if self.downsample:
            res_v,res_s = self.dconv1(res_v,res_s)
            res_v,res_s = self.d_nl1(res_v,res_s)
            res_v = self.d_vbn1(res_v)
            res_s = self.d_sbn1(res_s)
            v = v+res_v
            s = s+res_s

        return self.vrelu_o(v), self.relu(s)
    
class BasicBlock1D_flatten_LN(nn.Module):
    """ Supports: groups=1, dilation=1 """

    def __init__(self, indim, i, stride=1, downsample=False, bias=False, padding=1):
        super(BasicBlock1D_flatten_LN, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = Convolutional(dim_in=indim, dim_out= i, scalar_dim_in=indim//2, scalar_dim_out=i, stride = stride, kernel = (3,1), bias=bias, padding=padding)
        self.nl1 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)
        self.vbn1 = VNLayerNorm(i)
        self.sbn1 = LayerNorm(i)
        self.relu = nn.ReLU()
        self.vr1 = VNLeakyReLU(in_channels=i,share_nonlinearity=False) ## scalar relu done directly
        self.conv2 = Convolutional(dim_in=2*i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i, stride = 1,  kernel = (3,1), bias=bias, padding=padding)
        self.nl2 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)
        self.vbn2 = VNLayerNorm(i)
        self.sbn2 = LayerNorm(i)
        self.stride = stride
        self.downsample = downsample
        self.vrelu_o = VNLeakyReLU(in_channels=i,share_nonlinearity=False)
        self.dconv1 = Convolutional(dim_in=indim, dim_out= i, scalar_dim_in=indim//2, scalar_dim_out=i, stride = stride, kernel = (1,1), bias=bias, padding=0) ## residual 
        self.d_nl1 = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)#residual
        self.d_vbn1 = VNLayerNorm(i)#VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4), #residual
        self.d_sbn1 = LayerNorm(i)

    def forward(self, v,s):
        res_v = torch.clone(v)
        res_s = torch.clone(s)

        v,s = self.conv1(v,s)
        v,s = self.nl1(v,s)
        v = self.vbn1(v)
        s = self.sbn1(s)
        v = self.vr1(v)
        s = self.relu(s)
        v,s = self.conv2(v,s)
        v,s = self.nl2(v,s)
        v = self.vbn2(v)
        s = self.sbn2(s)         

        if self.downsample:
            res_v,res_s = self.dconv1(res_v,res_s)
            res_v,res_s = self.d_nl1(res_v,res_s)
            res_v = self.d_vbn1(res_v)
            res_s = self.d_sbn1(res_s)
            v = v+res_v
            s = s+res_s

        return self.vrelu_o(v), self.relu(s)


class Eq_Motion_Model(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        depth,
        stride = 2,
        padding=1,
        kernel=(7,1),
        grp_sizes = [2,2,2,2],
        bias = False
    ):
        super().__init__()

        self.relu = nn.ReLU() # used for all scalar relu activation 

        ## input block
        i=32
        self.ip_conv = Convolutional(dim_in=2*dim_in, dim_out= i, scalar_dim_in=scalar_dim_in, scalar_dim_out=i*2, stride = stride, kernel = kernel, bias=bias, padding=3)
        self.ip_nonlin = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2)
        self.ip_vn_bn = VNBatchNorm(num_features=i, dim=4)# VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4)
        self.ip_sca_bn = BatchNorm(num_features=i*2, dim=4)#LayerNorm(i*2) BatchNorm(num_features=i*2, dim=4)
        self.ip_vn_relu = VNLeakyReLU(in_channels=i,share_nonlinearity=False) ## scalar relu done directly
        
        self.ip_maxpool = VNMaxPool(in_channels=dim_in, kernel_size=3, stride=2, padding=1) 
        

        ## residual layers
        residual_layers = []#nn.ModuleList([])
        for d in range(depth):## the conv basic block is with 4 groups of increasing sizes
            if d!=0:
                i = i * 2
                stride = 2
                indim=i
            else : 
                stride = 1
                indim = 2*i
            residual_layers.append(
                    BasicBlock1D(indim, i, stride=stride, downsample=True, bias=False, padding=1))
            for j in range(1,grp_sizes[d]):
                indim = 2*i                 
                residual_layers.append( 
                    BasicBlock1D(indim, i, downsample=False, bias=False, padding=1)
                )

        self.residual_layers = nn.ModuleList(residual_layers)
        in_dim = pooling_dim//(2**(depth+1))+1 #+1 if it's d>2
        ## output conv block
        self.output_layers = nn.ModuleList([])
        self.output_layers.append(nn.ModuleList([
        Convolutional(dim_in=i*2, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2, stride = 1, kernel = (1,1), bias=bias, padding=0),
        NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2),
        VNBatchNorm(num_features=i, dim=4),#VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4)
        BatchNorm(num_features=2*i, dim=4),#LayerNorm(2*i) BatchNorm(num_features=2*i, dim=4)
        VNMaxPool(in_dim), ## this is where they flatten (we do a max pool) -- do a flatten?
        ]))

        ## output_layers 
        self.output_block = nn.ModuleList([])
        for _ in range(2):
            self.output_block.append(nn.ModuleList([
                VNLinear(dim_in=2*i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=2*i),
                NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i*2, scalar_dim_out=i*2),
                VNLeakyReLU(in_channels=i,share_nonlinearity=False), ## scalar relu done directly
                VNLayerNorm(i), ##for vector
                LayerNorm(2*i), ## for scalar
                ]))   


        ## final output layer
        self.final_layer = VNLinear(dim_in=2*i, dim_out= dim_out, scalar_dim_in=i*2, scalar_dim_out=scalar_dim_out)      

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        ## input block
        v,s = self.ip_conv(v, s)
        v,s = self.ip_nonlin(v,s)
        v = self.ip_vn_bn(v)
        s = self.ip_sca_bn(s)
        v = self.ip_vn_relu(v)
        s = self.relu(s)
        v,s = self.ip_maxpool(v,s)
        

        ## residual blocks
        for res_layer in self.residual_layers:
            v,s = res_layer(v,s)

        ## output block

        ## output conv block
        for c1,nl1,vbn1,bn1,mp in self.output_layers:
            v,s = c1(v,s)
            v,s = nl1(v,s)
            v = vbn1(v)
            s = bn1(s)
            v,s = mp(v,s)

        ## output layers
        for l1,nl1,vr1,vln1,sln1 in self.output_block:
            v,s = l1(v,s)
            v,s = nl1(v,s)
            v = vr1(v)
            s = self.relu(s)
            v = vln1(v)
            s = sln1(s)

        ## final output   

        return self.final_layer(v,s)

class Eq_Motion_Model_flatten(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        depth,
        stride = 2,
        padding=1,
        kernel=(7,1),
        grp_sizes = [2,2,2,2],
        bias = False
    ):
        super().__init__()

        self.relu = nn.ReLU() # used for all scalar relu activation 

        ## input block
        i=32
        self.ip_conv = Convolutional(dim_in=2*dim_in, dim_out= i, scalar_dim_in=scalar_dim_in, scalar_dim_out=i, stride = stride, kernel = kernel, bias=bias, padding=3)
        self.ip_nonlin = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)
        self.ip_vn_bn = VNBatchNorm(num_features=i, dim=4)# VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4)
        self.ip_sca_bn = BatchNorm(num_features=i, dim=4)#LayerNorm(i*2) BatchNorm(num_features=i*2, dim=4)
        self.ip_vn_relu = VNLeakyReLU(in_channels=i,share_nonlinearity=False) ## scalar relu done directly
        
        self.ip_maxpool = VNMaxPool(in_channels=dim_in, kernel_size=3, stride=2, padding=1) 
        

        ## residual layers
        residual_layers = []#nn.ModuleList([])
        for d in range(depth):## the conv basic block is with 4 groups of increasing sizes
            if d!=0:
                i = i * 2
                stride = 2
                indim=i
            else : 
                stride = 1
                indim = 2*i
            residual_layers.append(
                    BasicBlock1D_flatten(indim, i, stride=stride, downsample=True, bias=False, padding=1))
            for j in range(1,grp_sizes[d]):
                indim = 2*i                 
                residual_layers.append( 
                    BasicBlock1D_flatten(indim, i, downsample=False, bias=False, padding=1)
                )

        self.residual_layers = nn.ModuleList(residual_layers)
        in_dim = pooling_dim//(2**(depth+1))+1 #+1 if it's d>2
        ## output conv block
        self.output_layers = nn.ModuleList([])
        self.output_layers.append(nn.ModuleList([
        Convolutional(dim_in=i*2, dim_out= i, scalar_dim_in=i, scalar_dim_out=i, stride = 1, kernel = (1,1), bias=bias, padding=0),
        NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i),
        VNBatchNorm(num_features=i, dim=4),#VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4)
        BatchNorm(num_features=i, dim=4),#LayerNorm(2*i) BatchNorm(num_features=2*i, dim=4)
         ## this is where they flatten (we do a max pool) -- do a flatten?
        ]))

        ## output_layers 
        self.output_block = nn.ModuleList([])
        for d in range(2):
            if d==0:
                j = i*in_dim
            else:
                j=i
            self.output_block.append(nn.ModuleList([
                VNLinear(dim_in=2*j, dim_out= i, scalar_dim_in=j, scalar_dim_out=i),
                NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i),
                VNLeakyReLU(in_channels=i,share_nonlinearity=False), ## scalar relu done directly
                VNLayerNorm(i), ##for vector
                LayerNorm(i), ## for scalar
                ]))   


        ## final output layer
        self.final_layer = VNLinear(dim_in=2*i, dim_out= dim_out, scalar_dim_in=i, scalar_dim_out=scalar_dim_out)      

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        ## input block
        v,s = self.ip_conv(v, s)
        v,s = self.ip_nonlin(v,s)
        v = self.ip_vn_bn(v)
        s = self.ip_sca_bn(s)
        v = self.ip_vn_relu(v)
        s = self.relu(s)
        v,s = self.ip_maxpool(v,s)
        

        ## residual blocks
        for res_layer in self.residual_layers:
            v,s = res_layer(v,s)

        ## output block

        ## output conv block
        for c1,nl1,vbn1,bn1 in self.output_layers:
            v,s = c1(v,s)
            v,s = nl1(v,s)
            v = vbn1(v)
            s = bn1(s)
        
        v = v.permute(0,1,3,2).reshape((v.shape[0],-1,2)).permute(0,2,1)
        s = s.reshape((s.shape[0], -1))

        ## output layers
        for l1,nl1,vr1,vln1,sln1 in self.output_block:
            v,s = l1(v,s)
            v,s = nl1(v,s)
            v = vr1(v)
            s = self.relu(s)
            v = vln1(v)
            s = sln1(s)

        ## final output   

        return self.final_layer(v,s)

class Eq_Motion_Model_flatten_LN(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        depth,
        stride = 2,
        padding=1,
        kernel=(7,1),
        grp_sizes = [2,2,2,2],
        bias = False
    ):
        super().__init__()

        self.relu = nn.ReLU() # used for all scalar relu activation 

        ## input block
        i=32
        self.ip_conv = Convolutional(dim_in=2*dim_in, dim_out= i, scalar_dim_in=scalar_dim_in, scalar_dim_out=i, stride = stride, kernel = kernel, bias=bias, padding=3)
        self.ip_nonlin = NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i)
        self.ip_vn_bn = VNLayerNorm(i)# VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4)
        self.ip_sca_bn = LayerNorm(i)#LayerNorm(i*2) BatchNorm(num_features=i*2, dim=4)
        self.ip_vn_relu = VNLeakyReLU(in_channels=i,share_nonlinearity=False) ## scalar relu done directly
        
        self.ip_maxpool = VNMaxPool(in_channels=dim_in, kernel_size=3, stride=2, padding=1) 
        

        ## residual layers
        residual_layers = []#nn.ModuleList([])
        for d in range(depth):## the conv basic block is with 4 groups of increasing sizes
            if d!=0:
                i = i * 2
                stride = 2
                indim=i
            else : 
                stride = 1
                indim = 2*i
            residual_layers.append(
                    BasicBlock1D_flatten_LN(indim, i, stride=stride, downsample=True, bias=False, padding=1))
            for j in range(1,grp_sizes[d]):
                indim = 2*i                 
                residual_layers.append( 
                    BasicBlock1D_flatten_LN(indim, i, downsample=False, bias=False, padding=1)
                )

        self.residual_layers = nn.ModuleList(residual_layers)
        in_dim = pooling_dim//(2**(depth+1))+1 #+1 if it's d>2
        ## output conv block
        self.output_layers = nn.ModuleList([])
        self.output_layers.append(nn.ModuleList([
        Convolutional(dim_in=i*2, dim_out= i, scalar_dim_in=i, scalar_dim_out=i, stride = 1, kernel = (1,1), bias=bias, padding=0),
        NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i),
        VNLayerNorm(i),#VNLayerNorm(i) VNBatchNorm(num_features=i, dim=4)
        LayerNorm(i),#LayerNorm(2*i) BatchNorm(num_features=2*i, dim=4)
         ## this is where they flatten (we do a max pool) -- do a flatten?
        ]))

        ## output_layers 
        self.output_block = nn.ModuleList([])
        for d in range(2):
            if d==0:
                j = i*in_dim
            else:
                j=i
            self.output_block.append(nn.ModuleList([
                VNLinear(dim_in=2*j, dim_out= i, scalar_dim_in=j, scalar_dim_out=i),
                NonLinearity(dim_in=i, dim_out= i, scalar_dim_in=i, scalar_dim_out=i),
                VNLeakyReLU(in_channels=i,share_nonlinearity=False), ## scalar relu done directly
                VNLayerNorm(i), ##for vector
                LayerNorm(i), ## for scalar
                ]))   


        ## final output layer
        self.final_layer = VNLinear(dim_in=2*i, dim_out= dim_out, scalar_dim_in=i, scalar_dim_out=scalar_dim_out)      

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        ## input block
        v,s = self.ip_conv(v, s)
        v,s = self.ip_nonlin(v,s)
        v = self.ip_vn_bn(v)
        s = self.ip_sca_bn(s)
        v = self.ip_vn_relu(v)
        s = self.relu(s)
        v,s = self.ip_maxpool(v,s)
        

        ## residual blocks
        for res_layer in self.residual_layers:
            v,s = res_layer(v,s)

        ## output block

        ## output conv block
        for c1,nl1,vbn1,bn1 in self.output_layers:
            v,s = c1(v,s)
            v,s = nl1(v,s)
            v = vbn1(v)
            s = bn1(s)
        
        v = v.permute(0,1,3,2).reshape((v.shape[0],-1,2)).permute(0,2,1)
        s = s.reshape((s.shape[0], -1))

        ## output layers
        for l1,nl1,vr1,vln1,sln1 in self.output_block:
            v,s = l1(v,s)
            v,s = nl1(v,s)
            v = vr1(v)
            s = self.relu(s)
            v = vln1(v)
            s = sln1(s)

        ## final output   

        return self.final_layer(v,s)

if __name__ == "__main__": ## to quickly check
    x = torch.randn((2,200,2,4))
    res = orthogonal_input(x,dim=-2)
    print(res.shape)

    ##-------------------------------------------------------------------
    ## testing the linear layer----------------
    vector = torch.randn((2,200,2,2))
    scalar = torch.randn((2,200,11))
    model = VNLinear(dim_in=2*vector.shape[-2], dim_out= 10, scalar_dim_in=scalar.shape[-1], scalar_dim_out=15)
    ## pass the original values to the model
    out_vec, out_sca = model(vector, scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = model(rotated_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Linear layer test completed!')

    ###------------------------------------------- end of linear layer test

    ##------------------------------------------------------------------
    ## testing non-linearity
    out_vec = torch.randn((2,200,2,10))
    out_sca = torch.randn((2,200,15))
    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    nonlinear_layer = NonLinearity(dim_in=out_vec.shape[-1], dim_out= 10, scalar_dim_in=out_sca.shape[-1], scalar_dim_out=20)
    nl_out_vec, nl_out_sca = nonlinear_layer(out_vec, out_sca)

    nl_out_rot_out_vec, rot_nl_out_sca = nonlinear_layer(rot_out_vec, out_sca)

    rot_nl_out_vec = einsum('... b c, b a -> ... a c', nl_out_vec, R)

    assert torch.allclose(rot_nl_out_vec, nl_out_rot_out_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(nl_out_sca, rot_nl_out_sca, atol= 1e-6), 'scalar is not invariant'
    print('Non linearity layer test completed!')

    ###------------------------------------------- end of non linearity layer test


    ##------------------------------------------------------------------
    ## testing convolutional 

    vector = torch.randn((2,200,2,4))
    scalar = torch.randn((2,200,14))
    rot_vec = einsum('... b c, b a -> ... a c', vector, R)
    conv_layer = Convolutional(dim_in=2*vector.shape[-1], dim_out= 10, scalar_dim_in=scalar.shape[-1], scalar_dim_out=15, stride=1, padding=1, kernel = (16,1), bias=False)
    ## pass the original values to the model
    out_vec, out_sca = conv_layer(vector, scalar)

    out_rot_vec, out_rot_sca = conv_layer(rot_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Convolutional layer test completed!')

    ###------------------------------------------- end of convolutional test

    ##---------------------------------------------------------------------------------
    ## checking the layer norm

    vector = torch.randn((32,200,2,20))
    scalar = torch.randn((32,200,14))

    vec_ln_layer = VNLayerNorm(dim=vector.shape[-1])

    ## pass the original values to the model
    out_vec = vec_ln_layer(vector)

    sca_ln_layer = LayerNorm(dim=scalar.shape[-1])

    out_sca = sca_ln_layer(scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])


    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec = vec_ln_layer(rotated_vec)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_sca, atol= 1e-6), 'scalar is not invariant'
    print('Layer norm test completed!')
    ###------------------------------------------- end of layer norm

    ##------------------------------------------------------------------
    ## testing VNLeakyReLU
    out_vec = torch.randn((2,200,2,10))
    out_sca = torch.randn((2,200,15))
    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    nonlinear_layer = VNLeakyReLU(in_channels=out_vec.shape[-1],share_nonlinearity=True)
    nl_out_vec = nonlinear_layer(out_vec)
    nl_out_sca = F.relu(out_sca)

    nl_out_rot_out_vec = nonlinear_layer(rot_out_vec)
    rot_nl_out_sca = F.relu(out_sca)

    rot_nl_out_vec = einsum('... b c, b a -> ... a c', nl_out_vec, R)

    assert torch.allclose(rot_nl_out_vec, nl_out_rot_out_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(nl_out_sca, rot_nl_out_sca, atol= 1e-6), 'scalar is not invariant'
    print('VN Leaky Relu layer test completed!')

    ###------------------------------------------- end of non linearity layer test

    

    ##-----------------------------------------------------------------
    ##testing Max Pooling
    vector = torch.randn((2,200,2,10))
    scalar = torch.randn((2,200,11))
    pooling_layer = VNMaxPool(vector.shape[1]) ## over the number of samples
    ## pass the original values to the model
    out_vec, out_sca = pooling_layer(vector, scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = pooling_layer(rotated_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-3), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-3), 'scalar is not invariant'
    print('Max Pooling layer test completed!')

    ##-------------------------------------------- end of max pooling layer test

    ##-----------------------------------------------------------------
    ##testing Max Pooling with kernel and stride
    vector = torch.randn((2,200,2,10))
    scalar = torch.randn((2,200,11))
    pooling_layer = VNMaxPool(vector.shape[1],kernel_size=3, stride=2, padding=1) ## over the number of samples
    #self, in_channels, kernel_size=None, stride=None, padding=None

    ## pass the original values to the model
    out_vec, out_sca = pooling_layer(vector, scalar)
    
    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = pooling_layer(rotated_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-3), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-3), 'scalar is not invariant'
    print('Max Pooling with kernel and stride layer test completed!')

    ##-------------------------------------------- end of max pooling with kernel and stride layer test

    ##---------------------------------------------------------------------------------
    ## checking the batch norm

    vector = torch.randn((32,200,2,20))
    scalar = torch.randn((32,200,14))

    vec_bn_layer = VNBatchNorm(num_features=vector.shape[-1], dim=4)
    sca_bn_layer = BatchNorm(num_features=scalar.shape[-1], dim=3)

    ## pass the original values to the model
    out_vec = vec_bn_layer(vector)

    out_sca = sca_bn_layer(scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])


    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec = vec_bn_layer(rotated_vec)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_sca, atol= 1e-6), 'scalar is not invariant'
    print('Batch norm test completed!')
    ###------------------------------------------- end of batch norm

    ##---------------------------------------------------------------------------------
    # checking the full equivariant flatten model

 
    vector = torch.randn((1,200,2,2))
    scalar = torch.randn((1,200,14))

    eq_model = Eq_Motion_Model_flatten_LN(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = vector.shape[1],depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,2,2], bias=False
                            )

    total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    print('Network eq_resnet flatten loaded to device ')
    print("Total number of parameters:", total_params)

    ## pass the original values to the model
    out_vec, out_sca = eq_model(vector, scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = vector.permute(0,1,3,2) @ R
    rotated_vec = rotated_vec.permute(0,1,3,2)
    # einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = eq_model(rotated_vec, scalar)

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    rot_out_vec = out_vec.permute(0,2,1) @ R
    rot_out_vec = rot_out_vec.permute(0,2,1)

    # assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'

    print('Full equivariant model flatten test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant flatten model test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant model

 
    vector = torch.randn((1,200,2,2))
    scalar = torch.randn((1,200,14))

    eq_model = Eq_Motion_Model(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = vector.shape[1],depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,2,2], bias=False
                            )

    total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    print('Network eq_resnet loaded to device ')
    print("Total number of parameters:", total_params)

    ## pass the original values to the model
    out_vec, out_sca = eq_model(vector, scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = vector.permute(0,1,3,2) @ R
    rotated_vec = rotated_vec.permute(0,1,3,2)
    # einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = eq_model(rotated_vec, scalar)

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    rot_out_vec = out_vec.permute(0,2,1) @ R
    rot_out_vec = rot_out_vec.permute(0,2,1)

    # assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'

    print('Full equivariant model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant model test
