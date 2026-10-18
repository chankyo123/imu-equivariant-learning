import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from network.model_resnet import BasicBlock1D, ResNet1D 

# helper 

def orthogonal_input(x, dim=-1):
    return torch.concatenate((-x[..., 1,:].unsqueeze(-2), x[..., 0,:].unsqueeze(-2)), dim=dim)


class Eq_Motion_Model(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        tlio_in_dim,
        tlio_out_dim,
        tlio_depths, 
        tlio_net_config_in_dim,
        tlio_cov_dim_out=2,
    ):
        super().__init__()
        self.n_scalars = tlio_cov_dim_out


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
        v = torch.clone(input)
        v = v.permute(0,2,1)
        v = torch.cat((torch.cat((v[..., 0].unsqueeze(-1), v[..., 1].unsqueeze(-1)), dim=-1), torch.cat((v[..., 3].unsqueeze(-1), v[..., 4].unsqueeze(-1)), dim=-1)), dim=-2)

        # PCA to get the the first two principal components
        _,_,v = torch.pca_lowrank(v, center=True, niter = 3) # (B,2,2)
        v = v[:,:2,:2]

        v1 = v[...,0]/torch.norm(v[...,0], dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        v2 = v[...,1] - (v[...,1].unsqueeze(-2)@v1.unsqueeze(-1)).squeeze(-1)*v1

        #v2 = v[...,[1]] - (v[...,1].unsqueeze(-2)@v1)*v1
        v2 = v2/torch.norm(v2, dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        frame = torch.stack([v1,v2],dim=-1)
        
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
    network =  Eq_Motion_Model(tlio_in_dim=6, tlio_out_dim=3,tlio_depths=[2,2,2,2], 
                               tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=2)
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network loaded to device ')
    print("Total number of parameters:", total_params)
    frame, pred, cov = network(input)

    ## full cov
    network =  Eq_Motion_Model(tlio_in_dim=6, tlio_out_dim=3,tlio_depths=[2,2,2,2],
                               tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=3)
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network loaded to device ')
    print("Total number of parameters:", total_params)
    frame, pred, cov = network(input)

 
   