import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from network.model_resnet import BasicBlock1D, ResNet1D 
from fvcore.nn import FlopCountAnalysis
import time
from network.timing import CudaTimer

# helper 

def orthogonal_input(x, dim=-1):
    return torch.concatenate((-x[..., 1,:].unsqueeze(-2), x[..., 0,:].unsqueeze(-2)), dim=dim)

## equivariant modules

# class VNLinear(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_out,
#         scalar_dim_in,
#         scalar_dim_out,
#     ):
#         super().__init__()

#         self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)
#         self.scalar_linear = nn.Linear(scalar_dim_in, scalar_dim_out, bias=False)

#     def forward(self, vector, scalar):
#         return self.vector_linear(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1)), self.scalar_linear(scalar)

class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
    ):
        super().__init__()

        self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, vector):
        return self.vector_linear(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1))
    
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
        x = nn.ReLU()(x)
        x = self.layer_norm(x)
        if self.scalar_dim_out == 0:
            return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2))
        return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2)), x[..., -self.scalar_dim_out:]
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6): # dim is the vector dimension (i.e., 2)
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -2)
        x = x / norms.clamp(min = self.eps).unsqueeze(-2)
        return x * self.ln(norms).unsqueeze(-2)
    
class MeanPooling_layer(nn.Module):
    def __init__(
        self,
        dim = 1
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, vector, scalar):
        return torch.mean(vector, dim=self.dim), torch.mean(scalar, dim=self.dim)
    
class Convolutional(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()
        self.conv_layer_vec = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, stride=stride, kernel_size=kernel, padding=padding, bias=bias, padding_mode='replicate')
        self.conv_layer_sca = nn.Conv2d(in_channels=scalar_dim_in, out_channels=scalar_dim_out, stride=stride, kernel_size=kernel, padding=padding, bias=bias, padding_mode='replicate')

        
    def forward(self, vector, scalar):
        return self.conv_layer_vec(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1).permute(0,3,1,2)).permute(0,2,3,1), self.conv_layer_sca(scalar.unsqueeze(-2).permute(0,3,1,2)).permute(0,2,3,1).squeeze(-2)
    

class Eq_Motion_Model(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        tlio_in_dim,
        tlio_out_dim,
        tlio_depths, 
        tlio_net_config_in_dim,
        tlio_cov_dim_out=2,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.vnlinear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim)
        self.slinear_layer0 = nn.Linear(scalar_dim_in,scalar_hidden_dim, bias=False)
        self.nonlinearity0 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim, stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))


        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.vnlinear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)
        self.slinear_layer1 = nn.Linear(scalar_hidden_dim,scalar_hidden_dim, bias=False)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=0)
        self.vnlinear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)

        ##layer normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        
        ## output layer
        self.vnoutput_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out)

        ## TLIO
        self.tlio = ResNet1D(BasicBlock1D, tlio_in_dim, tlio_out_dim, tlio_depths, tlio_net_config_in_dim, cov_output_dim = tlio_cov_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, vector, scalar, original_scalars):
        v = torch.clone(vector)
        s = torch.clone(scalar)
        v = self.vnlinear_layer0(v)
        s = self.slinear_layer0(s)
        v,s = self.nonlinearity0(v,s)

        ## conv blocks
        for conv, nl, vnln, sln in self.layers:
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v,s = self.pooling_layer1(v,s)

        v = self.vnlinear_layer1(v)
        s = self.slinear_layer1(s)
        v = self.nonlinearity1(v,s) ## no scalar
        v = self.vnlinear_layer2(v)

        ##replace later with batch norm
        v = self.vector_ln1(v)

        v = self.vnoutput_layer(v)

        v1 = v[...,0]/torch.norm(v[...,0], dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        v2 = v[...,1] - (v[...,1].unsqueeze(-2)@v1.unsqueeze(-1)).squeeze(-1)*v1

        #v2 = v[...,[1]] - (v[...,1].unsqueeze(-2)@v1)*v1
        v2 = v2/torch.norm(v2, dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        frame = torch.stack([v1,v2],dim=-1)
        
        frame = frame.permute(0,2,1)
        v = torch.matmul(frame.unsqueeze(1),vector)

        input = torch.concat([v.reshape((*v.shape[:2],-1)), original_scalars], dim=-1).permute(0,2,1)

        disp, cov = self.tlio(input) ## change TLIO to 2 scalars
        #disp_inv = disp.clone()

        disp = torch.concat([torch.matmul(frame.permute(0,2,1), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)

        return frame, disp, cov #, disp_inv

class Eq_Motion_Model_fullCov(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        tlio_in_dim,
        tlio_out_dim,
        tlio_depths, 
        tlio_net_config_in_dim,
        tlio_cov_dim_out=2,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.vnlinear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim)
        self.slinear_layer0 = nn.Linear(scalar_dim_in,scalar_hidden_dim, bias=False)
        self.nonlinearity0 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim, stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))


        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.vnlinear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)
        self.slinear_layer1 = nn.Linear(scalar_hidden_dim,scalar_hidden_dim, bias=False)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=0)
        self.vnlinear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)

        ##layer normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        
        ## output layer
        self.vnoutput_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out)

        ## TLIO
        self.tlio = ResNet1D(BasicBlock1D, tlio_in_dim, tlio_out_dim, tlio_depths, tlio_net_config_in_dim, cov_output_dim = tlio_cov_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, vector, scalar, original_scalars):
        v = torch.clone(vector)
        s = torch.clone(scalar)
        v = self.vnlinear_layer0(v)
        s = self.slinear_layer0(s)
        v,s = self.nonlinearity0(v,s)

        ## conv blocks
        for conv, nl, vnln, sln in self.layers:
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v,s = self.pooling_layer1(v,s)

        v = self.vnlinear_layer1(v)
        s = self.slinear_layer1(s)
        v = self.nonlinearity1(v,s) ## we get only vector
        v = self.vnlinear_layer2(v)

        ##replace later with batch norm
        v = self.vector_ln1(v)

        v = self.vnoutput_layer(v)

        v1 = v[...,0]/torch.norm(v[...,0], dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        v2 = v[...,1] - (v[...,1].unsqueeze(-2)@v1.unsqueeze(-1)).squeeze(-1)*v1

        #v2 = v[...,[1]] - (v[...,1].unsqueeze(-2)@v1)*v1
        v2 = v2/torch.norm(v2, dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        frame = torch.stack([v1,v2],dim=-1)
        
        frame = frame.permute(0,2,1)
        v = torch.matmul(frame.unsqueeze(1),vector)

        input = torch.concat([v.reshape((*v.shape[:2],-1)), original_scalars], dim=-1).permute(0,2,1)

        disp, cov = self.tlio(input) 

        # disp = torch.concat([torch.matmul(frame.inverse(), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)

        # cov = torch.exp(2*cov)
        # new_cov = torch.matmul(frame.inverse(),torch.matmul(torch.diag_embed(cov[:,:-1], offset=0, dim1=-2, dim2=-1),frame))
        # new_cov = torch.cat((new_cov, torch.zeros((*new_cov.shape[:-1],1), device = new_cov.device)),dim=-1) 
        # new_cov = torch.cat((new_cov,torch.diag_embed(cov[:,-1].unsqueeze(-1), offset=2, dim1=-2, dim2=-1)[:,0,:].unsqueeze(-2)),dim=-2)   

        return frame, disp, cov
    
class Eq_Motion_Model_PearsonCov(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        tlio_in_dim,
        tlio_out_dim,
        tlio_depths, 
        tlio_net_config_in_dim,
        tlio_cov_dim_out=6,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.vnlinear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim)
        self.slinear_layer0 = nn.Linear(scalar_dim_in,scalar_hidden_dim, bias=False)
        self.nonlinearity0 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim, stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))


        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.vnlinear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)
        self.slinear_layer1 = nn.Linear(scalar_hidden_dim,scalar_hidden_dim, bias=False)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=0)
        self.vnlinear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)

        ##layer normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        
        ## output layer
        self.vnoutput_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out)

        ## TLIO
        self.tlio = ResNet1D(BasicBlock1D, tlio_in_dim, tlio_out_dim, tlio_depths, tlio_net_config_in_dim, cov_output_dim = tlio_cov_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, vector, scalar, original_scalars):
        v = torch.clone(vector)
        s = torch.clone(scalar)
        v = self.vnlinear_layer0(v)
        s = self.slinear_layer0(s)
        v,s = self.nonlinearity0(v,s)

        ## conv blocks
        for conv, nl, vnln, sln in self.layers:
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v,s = self.pooling_layer1(v,s)

        v = self.vnlinear_layer1(v)
        s = self.slinear_layer1(s)
        v = self.nonlinearity1(v,s) ## we get only vector
        v = self.vnlinear_layer2(v)

        ##replace later with batch norm
        v = self.vector_ln1(v)

        v = self.vnoutput_layer(v)

        v1 = v[...,0]/torch.norm(v[...,0], dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        v2 = v[...,1] - (v[...,1].unsqueeze(-2)@v1.unsqueeze(-1)).squeeze(-1)*v1

        #v2 = v[...,[1]] - (v[...,1].unsqueeze(-2)@v1)*v1
        v2 = v2/torch.norm(v2, dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        frame = torch.stack([v1,v2],dim=-1)
        
        frame = frame.permute(0,2,1)
        v = torch.matmul(frame.unsqueeze(1),vector)

        input = torch.concat([v.reshape((*v.shape[:2],-1)), original_scalars], dim=-1).permute(0,2,1)

        disp, p = self.tlio(input) 
        # cov = torch.eye(3).expand(disp.shape[0],-1,-1).reshape((-1,9))
        # "force the Pearson correlation coefficients to not get too close to 1"
        cov = torch.empty((disp.shape[0], 9)).to(p.device)
        MIN_LOG_STD = torch.log(torch.Tensor([1e-2]).to(p.device))
        # print('before:',torch.cuda.memory_allocated(0))
        # on diagonal terms
        
        a = torch.exp(2 * torch.maximum(p[:, 0], MIN_LOG_STD))
        b = torch.exp(2 * torch.maximum(p[:, 1], MIN_LOG_STD))
        c = torch.exp(2 * torch.maximum(p[:, 2], MIN_LOG_STD))

        cov = torch.stack((a, 
                           (1 - 1e-3) * torch.tanh(0.05 * p[:, 3]) * torch.sqrt(a * b),
                           (1 - 1e-3) * torch.tanh(0.05 * p[:, 4]) * torch.sqrt(a * c), 
                           (1 - 1e-3) * torch.tanh(0.05 * p[:, 3]) * torch.sqrt(a * b),
                           b,
                           (1 - 1e-3) * torch.tanh(0.05 * p[:, 5]) * torch.sqrt(b * c),
                           (1 - 1e-3) * torch.tanh(0.05 * p[:, 4]) * torch.sqrt(a * c),
                           (1 - 1e-3) * torch.tanh(0.05 * p[:, 5]) * torch.sqrt(b * c),
                           c),dim=-1)
        # print(cov.shape)
        # off diagonal terms
        # cov[:, 1] = (1 - 1e-3) * torch.tanh(0.05 * p[:, 3]) * torch.sqrt(cov[:, 0] * cov[:, 4])  # xy
        # cov[:, 2] = ((1 - 1e-3) * torch.tanh(0.05 * p[:, 4]) * torch.sqrt(cov[:, 0] * cov[:, 8])).float()  # xz
        # cov[:, 5] = ((1 - 1e-3) * torch.tanh(0.05 * p[:, 5]) * torch.sqrt(cov[:, 4] * cov[:, 8])).float()  # yz
        # # symmetry
        # cov[:, 3] = cov[:, 1] 
        # cov[:, 6] = cov[:, 2]  # xy
        # cov[:, 7] = cov[:, 5]  # xy


        # print('after:',torch.cuda.memory_allocated(0))
        return frame, disp, cov.reshape((disp.shape[0], 3, 3))
    
if __name__ == "__main__": ## to quickly check
    # eq_model = Eq_Motion_Model_fullCov(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
    #                            tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=3,
    #                            hidden_dim=128, scalar_hidden_dim=128, depth=1, stride = 1, padding='same', kernel =(32,1), bias=False
    #                         )
    # eq_model.eval()
    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network Eq_Motion_Model_fullCov loaded to device ')
    # print("Total number of parameters:", total_params)
    # x = torch.randn((2,200,2,4))
    # res = orthogonal_input(x,dim=-2)
    # print(res.shape)


    # ##-------------------------------------------------------------------
    # ## testing the linear layer----------------
    # vector = torch.randn((2,200,2,2))
    # scalar = torch.randn((2,200,11))
    # model_v = VNLinear(dim_in=2*vector.shape[-2], dim_out= 10)
    # model_s = nn.Linear(scalar.shape[-1], 15, bias=False)
    # ## pass the original values to the model
    # out_vec = model_v(vector)
    # out_sca = model_s(scalar)

    # yaw = torch.randn((1))
    # R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    # rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    # out_rot_vec = model_v(rotated_vec)
    # out_rot_sca = model_s(scalar)

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    # assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    # print('Linear layer test completed!')

    # ###------------------------------------------- end of linear layer test

    # ##------------------------------------------------------------------
    # ## testing non-linearity
    # out_vec = torch.randn((2,200,2,10))
    # out_sca = torch.randn((2,200,15))
    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    # nonlinear_layer = NonLinearity(dim_in=out_vec.shape[-1], dim_out= 10, scalar_dim_in=out_sca.shape[-1], scalar_dim_out=20)
    # nl_out_vec, nl_out_sca = nonlinear_layer(out_vec, out_sca)

    # nl_out_rot_out_vec, rot_nl_out_sca = nonlinear_layer(rot_out_vec, out_sca)

    # rot_nl_out_vec = einsum('... b c, b a -> ... a c', nl_out_vec, R)

    # assert torch.allclose(rot_nl_out_vec, nl_out_rot_out_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(nl_out_sca, rot_nl_out_sca, atol= 1e-6), 'scalar is not invariant'
    # print('Non linearity layer test completed!')

    # ###------------------------------------------- end of non linearity layer test

    # ##---------------------------------------------------------------------------------
    # ## checking the layer norm

    # vector = torch.randn((32,200,2,20))
    # scalar = torch.randn((32,200,14))

    # vec_ln_layer = VNLayerNorm(dim=vector.shape[-1])

    # ## pass the original values to the model
    # out_vec = vec_ln_layer(vector)

    # sca_ln_layer = LayerNorm(dim=scalar.shape[-1])

    # out_sca = sca_ln_layer(scalar)

    # yaw = torch.randn((1))
    # R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])


    # rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    # out_rot_vec = vec_ln_layer(rotated_vec)

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    # assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_sca, atol= 1e-6), 'scalar is not invariant'
    # print('Layer norm test completed!')
    # ###------------------------------------------- end of layer norm

    # ##-----------------------------------------------------------------
    # ##testing Mean Pooling
    # vector = torch.randn((2,200,2,10))
    # scalar = torch.randn((2,200,11))
    # pooling_layer = MeanPooling_layer(dim = 1) ## over the number of samples
    # ## pass the original values to the model
    # out_vec, out_sca = pooling_layer(vector, scalar)

    # yaw = torch.randn((1))
    # R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    # rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    # out_rot_vec, out_rot_sca = pooling_layer(rotated_vec, scalar)

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    # assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    # print('Mean Pooling layer test completed!')

    # ##-------------------------------------------- end of mean pooling layer test

    # ##------------------------------------------------------------------
    # ## testing convolutional 

    # vector = torch.randn((2,200,2,4))
    # scalar = torch.randn((2,200,14))
    # rot_vec = einsum('... b c, b a -> ... a c', vector, R)
    # conv_layer = Convolutional(dim_in=2*vector.shape[-1], dim_out= 10, scalar_dim_in=scalar.shape[-1], scalar_dim_out=15, stride=1, padding="same", kernel = (16,1), bias=False)
    # ## pass the original values to the model
    # out_vec, out_sca = conv_layer(vector, scalar)

    # out_rot_vec, out_rot_sca = conv_layer(rot_vec, scalar)

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    # assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    # print('Convolutional layer test completed!')

    ###------------------------------------------- end of convolutional test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant model

 
    # vector = torch.randn((1,200,2,2)).to('cuda')
    # scalar = torch.randn((1,200,5)).to('cuda')
    # original_scalars = torch.randn((1,200,2)).to('cuda')

    
    eq_model = Eq_Motion_Model_fullCov(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=3,
                               hidden_dim=128, scalar_hidden_dim=128, depth=1, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    eq_model.eval()
    eq_model = eq_model.to('cuda')
    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network Eq_Motion_Model_fullCov loaded to device ')
    # print("Total number of parameters:", total_params)

    # flops = FlopCountAnalysis(eq_model, (vector, scalar, original_scalars))
    # total_flops = flops.total()
    # print(f"FLOPs: {total_flops}")

    # # Measure inference time
    # start_time = time.time()
    # frame, disp, cov = eq_model(vector, scalar, original_scalars)
    # end_time = time.time()
    # inference_time = end_time - start_time

    # print(f"Inference Time: {inference_time} seconds")
    #eq_model = torch.compile(eq_model)
    total_flops = 0
    inference_time = 0
    allocated_memory = 0
    memory = 0
    for i in range(100):
        vector = torch.randn((1,200,2,2)).to('cuda')
        scalar = torch.randn((1,200,5)).to('cuda')
        original_scalars = torch.randn((1,200,2)).to('cuda')
        print(vector.device)
        flops = FlopCountAnalysis(eq_model, (vector, scalar, original_scalars))

        # end_time = time.time()
        if i>5:
            # inference_time += end_time - start_time
            total_flops += flops.total()
            allocated_memory += torch.cuda.max_memory_allocated('cuda') / 1024**2

            # Measure inference time
            # start_time = time.time()
            with CudaTimer("SO2"):
                frame, disp, cov = eq_model(vector, scalar, original_scalars)
        else:
            frame, disp, cov = eq_model(vector, scalar, original_scalars)

        new_mem, _ = torch.cuda.mem_get_info()
        memory +=new_mem/1024**2
    CudaTimer.print_timings()
        
    print(f"FLOPs: {total_flops/10}")
    # print(f"Inference Time: {inference_time/10} seconds")
    print('memory:', allocated_memory/10)
    print('mem info:', memory/10)
    print(f"FLOPs: {total_flops/95}")
    # print(f"Inference Time: {inference_time/10} seconds")
    # Calculate TOPs required
    # TOPs = total_flops / 1e12 / inference_time
    # print(f"TOPs required: {TOPs}")

    # ## pass the original values to the model
    # frame, disp, cov = eq_model(vector, scalar, original_scalars)
    # out_d = torch.concat([torch.matmul(frame.permute(0,2,1), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)

    # cov = torch.exp(2*cov)
    # out_c = torch.matmul(frame.permute(0,2,1),torch.matmul(torch.diag_embed(cov[:,:-1], offset=0, dim1=-2, dim2=-1),frame))
    # out_c = torch.cat((out_c, torch.zeros((*out_c.shape[:-1],1), device = out_c.device)),dim=-1) 
    # out_c = torch.cat((out_c,torch.diag_embed(cov[:,-1].unsqueeze(-1), offset=2, dim1=-2, dim2=-1)[:,0,:].unsqueeze(-2)),dim=-2)   

    # yaw = torch.randn((1))
    # R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    # rotated_vec = vector.permute(0,1,3,2) @ R
    # rotated_vec = rotated_vec.permute(0,1,3,2)
    # # einsum('... b c, b a -> ... a c', vector, R)
    # frame, disp, cov = eq_model(rotated_vec, scalar, original_scalars)
    # out_rot_d = torch.concat([torch.matmul(frame.permute(0,2,1), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)

    # cov = torch.exp(2*cov)
    # out_rot_c = torch.matmul(frame.permute(0,2,1),torch.matmul(torch.diag_embed(cov[:,:-1], offset=0, dim1=-2, dim2=-1),frame))
    # out_rot_c = torch.cat((out_rot_c, torch.zeros((*out_rot_c.shape[:-1],1), device = out_rot_c.device)),dim=-1) 
    # out_rot_c = torch.cat((out_rot_c,torch.diag_embed(cov[:,-1].unsqueeze(-1), offset=2, dim1=-2, dim2=-1)[:,0,:].unsqueeze(-2)),dim=-2)   


    # # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    # R_o = torch.Tensor([[torch.cos(yaw), torch.sin(yaw), 0], [-torch.sin(yaw), torch.cos(yaw), 0], [0,0,1]])
    # rot_out_d = torch.matmul(R_o.unsqueeze(0),out_d.unsqueeze(-1)).squeeze(-1)
    # rot_out_c = torch.matmul(R_o.unsqueeze(0),torch.matmul(out_c,R_o.T))
    

    # assert torch.allclose(rot_out_d, out_rot_d, atol = 1e-4), 'vector is not equivariant'
    # assert torch.allclose(rot_out_c, out_rot_c, atol= 1e-4), 'covariance is not equivariant'

    # print('Full Eq_Motion_Model_fullCov model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant model test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant model

 
    # vector = torch.randn((2,200,2,2))
    # scalar = torch.randn((2,200,5))
    # original_scalars = torch.randn((2,200,2))

    
    # eq_model = Eq_Motion_Model(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
    #                            tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=2,
    #                            hidden_dim=128, scalar_hidden_dim=128, depth=1, stride = 1, padding='same', kernel =(32,1), bias=False
    #                         )
    # eq_model.eval()
    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq frame 2 scalars loaded to device ')
    # print("Total number of parameters:", total_params)

    # ## pass the original values to the model
    # frame, out_d, out_c = eq_model(vector, scalar, original_scalars)#, d_inv

    # yaw = torch.randn((1))
    # R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    # rotated_vec = vector.permute(0,1,3,2) @ R
    # rotated_vec = rotated_vec.permute(0,1,3,2)
    # # einsum('... b c, b a -> ... a c', vector, R)
    # frame,out_rot_d, out_rot_c = eq_model(rotated_vec, scalar, original_scalars) #, d_inv

    # # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    # R_o = torch.Tensor([[torch.cos(yaw), torch.sin(yaw), 0], [-torch.sin(yaw), torch.cos(yaw), 0], [0,0,1]])
    # rot_out_d = torch.matmul(R_o.unsqueeze(0),out_d.unsqueeze(-1)).squeeze(-1)

    # assert torch.allclose(rot_out_d, out_rot_d, atol = 1e-4), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-4), 'covariance is not invariant'

    # print('Full equivariant model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant model test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant model

 
    # vector = torch.randn((2,200,2,2))
    # scalar = torch.randn((2,200,5))
    # original_scalars = torch.randn((2,200,2))

    
    # eq_model = Eq_Motion_Model_PearsonCov(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
    #                            tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=6,
    #                            hidden_dim=128, scalar_hidden_dim=128, depth=1, stride = 1, padding='same', kernel =(32,1), bias=False
    #                         )
    # eq_model.eval()
    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq frame pearson cov loaded to device ')
    # print("Total number of parameters:", total_params)

    # ## pass the original values to the model
    # frame, disp, out_c = eq_model(vector, scalar, original_scalars)#, d_inv
    # out_d = torch.concat([torch.matmul(frame.permute(0,2,1), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)
    # frame_3d = torch.cat((torch.cat((frame, torch.zeros(*frame.shape[:-1], 1)), dim=-1), torch.Tensor([0,0,1]).reshape(1,1,3).broadcast_to(frame.shape[0],1,3)), dim=-2)
    # out_c = torch.matmul(frame_3d.permute(0,2,1),torch.matmul(out_c,frame_3d))

    # yaw = torch.randn((1))
    # R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    # rotated_vec = vector.permute(0,1,3,2) @ R
    # rotated_vec = rotated_vec.permute(0,1,3,2)
    # # einsum('... b c, b a -> ... a c', vector, R)
    # frame,disp, out_rot_c = eq_model(rotated_vec, scalar, original_scalars) #, d_inv
    # out_rot_d = torch.concat([torch.matmul(frame.permute(0,2,1), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)
    # frame_3d = torch.cat((torch.cat((frame, torch.zeros(*frame.shape[:-1], 1)), dim=-1), torch.Tensor([0,0,1]).reshape(1,1,3).broadcast_to(frame.shape[0],1,3)), dim=-2)
    # out_rot_c = torch.matmul(frame_3d.permute(0,2,1),torch.matmul(out_rot_c,frame_3d))

    # # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    # R_o = torch.Tensor([[torch.cos(yaw), torch.sin(yaw), 0], [-torch.sin(yaw), torch.cos(yaw), 0], [0,0,1]])
    # rot_out_d = torch.matmul(R_o.unsqueeze(0),out_d.unsqueeze(-1)).squeeze(-1)
    # rot_out_c = torch.matmul(R_o.unsqueeze(0),torch.matmul(out_c,R_o.T))

    # assert torch.allclose(rot_out_d, out_rot_d, atol = 1e-4), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-4), 'covariance is not equivariant'

    # print('Full equivariant model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant model test

 
   