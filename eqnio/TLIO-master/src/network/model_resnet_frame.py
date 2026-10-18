import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from network.model_resnet import BasicBlock1D, ResNet1D 

# helper 

def orthogonal_input(x, dim=-1):
    return torch.concatenate((-x[..., 1,:].unsqueeze(-2), x[..., 0,:].unsqueeze(-2)), dim=dim)

## non equivariant modules

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class MeanPooling_layer(nn.Module):
    def __init__(
        self,
        dim = -1
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, scalar):
        return torch.mean(scalar, dim=self.dim)

class Eq_Motion_Model(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        frame_dim_in,
        dim_out,
        frame_hidden_dim ,  
        depth,
        tlio_in_dim,
        tlio_out_dim,
        tlio_depths, 
        pooling_dim,
        tlio_net_config_in_dim,
        tlio_cov_dim_out=2,
        stride = 1,
        padding='same',
        kernel=16,
        bias = False
    ):
        super().__init__()
        self.n_scalars = tlio_cov_dim_out

        self.linear_layer0 = nn.Linear(frame_dim_in,frame_hidden_dim, bias=bias)
        self.relu = nn.ReLU()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                nn.Conv1d(frame_hidden_dim,frame_hidden_dim,kernel_size=kernel,stride=stride,padding=padding,bias=False,padding_mode='replicate'),
                ## relu
                LayerNorm(frame_hidden_dim), 
            ]))


        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.linear_layer1 = nn.Linear(frame_hidden_dim, frame_hidden_dim, bias=bias)
        # relu
        self.linear_layer2 = nn.Linear(frame_hidden_dim, frame_hidden_dim, bias=bias)

        ##layer normalization
        self.ln1 = LayerNorm(frame_hidden_dim) 
        ## output layer
        self.output_layer = nn.Linear(frame_hidden_dim, dim_out, bias=bias)

        ## TLIO
        self.tlio = ResNet1D(BasicBlock1D, tlio_in_dim, tlio_out_dim, tlio_depths, tlio_net_config_in_dim, cov_output_dim = tlio_cov_dim_out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input):
        ## tlio input is of the form (1024,6,200)
        original_scalars = torch.cat((input[...,5,:].unsqueeze(-2), input[...,2,:].unsqueeze(-2)), dim=-2)
        vector = torch.clone(input)
        vector = vector.permute(0,2,1)
        vector = torch.cat((torch.cat((vector[..., 0].unsqueeze(-1), vector[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2), torch.cat((vector[..., 3].unsqueeze(-1), vector[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)), dim=-2)
        v = input.permute(0,2,1)
        v = self.linear_layer0(v)
        v = self.relu(v)

        ## conv blocks
        for conv, ln in self.layers:
            v = conv(v.permute(0,2,1)).permute(0,2,1)
            v = self.relu(v)
            v = ln(v)

        v = self.pooling_layer1(v)

        v = self.linear_layer1(v)
        v = self.relu(v)
        v = self.linear_layer2(v)

        ##replace later with batch norm
        v = self.ln1(v)

        v = self.output_layer(v)
        v = v.unsqueeze(-1)

        frame = torch.concat([v/torch.norm(v, dim=-2).clamp(min = 1e-6).unsqueeze(-2),orthogonal_input(v/torch.norm(v, dim=-2).clamp(min = 1e-6).unsqueeze(-2),dim=-2)], dim=-1)
        frame = frame.permute(0,2,1)
        v = torch.matmul(frame.unsqueeze(1),vector)

        input = torch.concat([v.reshape((*v.shape[:2],-1)).permute(0,2,1), original_scalars], dim=-2)

        disp, cov = self.tlio(input) ## change TLIO to 2 scalars
        #disp_inv = disp.clone()

        if self.n_scalars == 2:
            disp = torch.concat([torch.matmul(frame.permute(0,2,1), disp[:,:2].unsqueeze(-1)).squeeze(-1),disp[:,-1].unsqueeze(-1)], dim=-1)        

        return frame, disp, cov #, disp_inv

if __name__ == "__main__": ## to quickly check
    input = torch.randn((1024,6,200))
    network =  Eq_Motion_Model(frame_dim_in=6,dim_out=2,frame_hidden_dim=256 , depth=2,tlio_in_dim=6, pooling_dim=-2,
                               tlio_out_dim=3,tlio_depths=[2,2,2,2], tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=2,
                               stride = 1, padding='same', kernel=16,bias = False)
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network loaded to device ')
    print("Total number of parameters:", total_params)
    frame, pred, cov = network(input)

    ## full cov
    network =  Eq_Motion_Model(frame_dim_in=6,dim_out=2,frame_hidden_dim=256 , depth=2,tlio_in_dim=6, pooling_dim=-2,
                               tlio_out_dim=3,tlio_depths=[2,2,2,2], tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=3,
                               stride = 1, padding='same', kernel=16,bias = False)
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network loaded to device ')
    print("Total number of parameters:", total_params)
    frame, pred, cov = network(input)

 
   