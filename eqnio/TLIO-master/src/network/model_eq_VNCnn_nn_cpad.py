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
        # self.layer_norm = LayerNorm(dim_out+scalar_dim_out)

        
    def forward(self, vector, scalar):
        x = torch.concatenate((torch.norm(vector, dim=-2), scalar), dim=-1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        # x = self.layer_norm(x)
        return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2)), x[..., -self.scalar_dim_out:]
    
class NonLinearity_innerpdt(nn.Module):
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

        self.linear = nn.Linear((dim_in*dim_in)+scalar_dim_in, dim_out+scalar_dim_out, bias=False)
        self.layer_norm = LayerNorm(dim_out+scalar_dim_out)

        
    def forward(self, vector, scalar):
        if vector.dim()==4:
            x = torch.concatenate((torch.flatten(torch.matmul(vector.permute(0,1,3,2), vector), start_dim=2), scalar), dim=-1)
        elif vector.dim()==3:
            x = torch.concatenate((torch.flatten(torch.matmul(vector.permute(0,2,1), vector), start_dim=1), scalar), dim=-1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.layer_norm(x)
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
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.linear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim, scalar_dim_in=scalar_dim_in, scalar_dim_out=scalar_hidden_dim)
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
        self.linear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.linear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)

        ##batch normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        self.scalar_ln1 = LayerNorm(scalar_hidden_dim) ## for scalar
        
        ## output layer
        self.output_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        v,s = self.linear_layer0(v, s)
        v,s = self.nonlinearity0(v,s)

        ## transformer blocks
        for conv, nl, vnln, sln in self.layers:
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v,s = self.pooling_layer1(v,s)

        v,s = self.linear_layer1(v,s)
        v,s = self.nonlinearity1(v,s)
        v,s = self.linear_layer2(v,s)

        ##replace later with batch norm
        v = self.vector_ln1(v)
        s = self.scalar_ln1(s) 

        return self.output_layer(v,s)
    
class Eq_Motion_Model_flatten(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.linear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim, scalar_dim_in=scalar_dim_in, scalar_dim_out=scalar_hidden_dim)
        self.nonlinearity0 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim, stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))


        # self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.linear_layer1 = VNLinear(dim_in=2*hidden_dim*pooling_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim*pooling_dim, scalar_dim_out=scalar_hidden_dim)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.linear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)

        ##batch normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        self.scalar_ln1 = LayerNorm(scalar_hidden_dim) ## for scalar
        
        ## output layer
        self.output_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        v,s = self.linear_layer0(v, s)
        v,s = self.nonlinearity0(v,s)

        ## transformer blocks
        for conv, nl, vnln, sln in self.layers:
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v = v.permute(0,1,3,2).reshape((v.shape[0],-1,2)).permute(0,2,1)
        s = s.reshape((s.shape[0], -1))

        v,s = self.linear_layer1(v,s)
        v,s = self.nonlinearity1(v,s)
        v,s = self.linear_layer2(v,s)

        ##replace later with batch norm
        v = self.vector_ln1(v)
        s = self.scalar_ln1(s) 

        return self.output_layer(v,s)

class Eq_ResidualCNN_Motion_Model(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.linear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim, scalar_dim_in=scalar_dim_in, scalar_dim_out=scalar_hidden_dim)
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
        self.linear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.linear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)

        ##batch normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        self.scalar_ln1 = LayerNorm(scalar_hidden_dim) ## for scalar
        
        ## output layer
        self.output_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        v,s = self.linear_layer0(v, s)
        v,s = self.nonlinearity0(v,s)

        ## transformer blocks
        for conv, nl, vnln, sln in self.layers:
            res_v = v.clone()
            res_s = s.clone()
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v+res_v)
            s = sln(s+res_s)

        v,s = self.pooling_layer1(v,s)

        v,s = self.linear_layer1(v,s)
        v,s = self.nonlinearity1(v,s)
        v,s = self.linear_layer2(v,s)

        ##replace later with batch norm
        v = self.vector_ln1(v)
        s = self.scalar_ln1(s)
    

        return self.output_layer(v,s)

class Eq_ResidualCNN_nlip_Motion_Model(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.linear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim, scalar_dim_in=scalar_dim_in, scalar_dim_out=scalar_hidden_dim)
        self.nonlinearity0 = NonLinearity_innerpdt(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim, stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity_innerpdt(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))


        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.linear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.nonlinearity1 = NonLinearity_innerpdt(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        self.linear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)

        ##batch normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        self.scalar_ln1 = LayerNorm(scalar_hidden_dim) ## for scalar
        
        ## output layer
        self.output_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, v, s):
        v,s = self.linear_layer0(v, s)
        v,s = self.nonlinearity0(v,s)

        ## transformer blocks
        for conv, nl, vnln, sln in self.layers:
            res_v = v.clone()
            res_s = s.clone()
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v+res_v)
            s = sln(s+res_s)

        v,s = self.pooling_layer1(v,s)

        v,s = self.linear_layer1(v,s)
        v,s = self.nonlinearity1(v,s)
        v,s = self.linear_layer2(v,s)

        ##replace later with batch norm
        v = self.vector_ln1(v)
        s = self.scalar_ln1(s)
 

        return self.output_layer(v,s)

if __name__ == "__main__": ## to quickly check
    x = torch.randn((2,200,2,4))
    res = orthogonal_input(x,dim=-2)
    print(res.shape)

    ##------------------------------------------------------------------
    ## testing non-linearity inner pdt
    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])
    out_vec = torch.randn((2,200,2,10))
    out_sca = torch.randn((2,200,15))
    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    nonlinear_layer = NonLinearity_innerpdt(dim_in=out_vec.shape[-1], dim_out= 10, scalar_dim_in=out_sca.shape[-1], scalar_dim_out=20)
    nl_out_vec, nl_out_sca = nonlinear_layer(out_vec, out_sca)

    nl_out_rot_out_vec, rot_nl_out_sca = nonlinear_layer(rot_out_vec, out_sca)

    rot_nl_out_vec = einsum('... b c, b a -> ... a c', nl_out_vec, R)

    assert torch.allclose(rot_nl_out_vec, nl_out_rot_out_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(nl_out_sca, rot_nl_out_sca, atol= 1e-6), 'scalar is not invariant'
    print('Non linearity inner pdt layer test completed!')

    ###------------------------------------------- end of non linearity inner pdt layer test

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

    ##-----------------------------------------------------------------
    ##testing Mean Pooling
    vector = torch.randn((2,200,2,10))
    scalar = torch.randn((2,200,11))
    pooling_layer = MeanPooling_layer(dim = 1) ## over the number of samples
    ## pass the original values to the model
    out_vec, out_sca = pooling_layer(vector, scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = pooling_layer(rotated_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Mean Pooling layer test completed!')

    ##-------------------------------------------- end of mean pooling layer test

    ##------------------------------------------------------------------
    ## testing convolutional 

    vector = torch.randn((2,200,2,4))
    scalar = torch.randn((2,200,14))
    rot_vec = einsum('... b c, b a -> ... a c', vector, R)
    conv_layer = Convolutional(dim_in=2*vector.shape[-1], dim_out= 10, scalar_dim_in=scalar.shape[-1], scalar_dim_out=15, stride=1, padding="same", kernel = (16,1), bias=False)
    ## pass the original values to the model
    out_vec, out_sca = conv_layer(vector, scalar)

    out_rot_vec, out_rot_sca = conv_layer(rot_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Convolutional layer test completed!')

    ###------------------------------------------- end of convolutional test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant model

 
    vector = torch.randn((2,200,2,2))
    scalar = torch.randn((2,200,14))

    eq_model = Eq_Motion_Model_flatten(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 200, 
                               hidden_dim=128, scalar_hidden_dim=128, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False
                            )

    total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    print('Network eq_transformer loaded to device ')
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

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'

    print('Full equivariant model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant model test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant residual cnn model

 
    vector = torch.randn((2,20,2,2))
    scalar = torch.randn((2,20,14))

    eq_model = Eq_ResidualCNN_Motion_Model(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 1, 
                               hidden_dim=128, scalar_hidden_dim=128, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False
                            )

    total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    print('Network eq_transformer loaded to device ')
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

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'

    print('Full equivariant residual cnn model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant residual cnn model test

    ##---------------------------------------------------------------------------------
    # checking the full equivariant residual cnn nlip model

 
    vector = torch.randn((2,200,2,2))
    scalar = torch.randn((2,200,14))

    eq_model = Eq_ResidualCNN_nlip_Motion_Model(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 1, 
                               hidden_dim=64, scalar_hidden_dim=64, depth=8, stride = 1, padding='same', kernel =(8,1), bias=False
                            )

    total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    print('Network eq_transformer loaded to device ')
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

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'

    print('Full equivariant residual cnn nlip model test completed!')


    # total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    # print('Network eq_transformer loaded to device ')
    # print("Total number of parameters:", total_params)

    ###------------------------------------------- end of full equivariant residual cnn nlip model test
