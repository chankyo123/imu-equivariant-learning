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

        ### frame 1
        frame1 = torch.stack([v1,v2],dim=-1)
        
        frame1 = frame1.permute(0,2,1)
        v_1 = torch.matmul(frame1.unsqueeze(1),vector)

        input1 = torch.concat([v_1.reshape((*v_1.shape[:2],-1)).permute(0,2,1), original_scalars], dim=-2)

        disp1, cov1 = self.tlio(input1) ## change TLIO to 2 scalars
        #disp_inv = disp.clone()

        ### frame 2
        frame2 = torch.stack([-v1,v2],dim=-1)
        
        frame2 = frame2.permute(0,2,1)
        v_2 = torch.matmul(frame2.unsqueeze(1),vector)

        input2 = torch.concat([v_2.reshape((*v_2.shape[:2],-1)).permute(0,2,1), original_scalars], dim=-2)

        disp2, cov2 = self.tlio(input2) ## change TLIO to 2 scalars

        ### frame 3
        frame3 = torch.stack([v1,-v2],dim=-1)
        
        frame3 = frame3.permute(0,2,1)
        v_3 = torch.matmul(frame3.unsqueeze(1),vector)

        input3 = torch.concat([v_3.reshape((*v_3.shape[:2],-1)).permute(0,2,1), original_scalars], dim=-2)

        disp3, cov3 = self.tlio(input3) ## change TLIO to 2 scalars

        ### frame 4
        frame4 = torch.stack([-v1,-v2],dim=-1)
        
        frame4 = frame4.permute(0,2,1)
        v_4 = torch.matmul(frame4.unsqueeze(1),vector)

        input4 = torch.concat([v_4.reshape((*v_4.shape[:2],-1)).permute(0,2,1), original_scalars], dim=-2)

        disp4, cov4 = self.tlio(input4) ## change TLIO to 2 scalars

        if self.n_scalars == 2:
            disp1 = torch.concat([torch.matmul(frame1.permute(0,2,1), disp1[:,:2].unsqueeze(-1)).squeeze(-1),disp1[:,-1].unsqueeze(-1)], dim=-1) 
            disp2 = torch.concat([torch.matmul(frame2.permute(0,2,1), disp2[:,:2].unsqueeze(-1)).squeeze(-1),disp2[:,-1].unsqueeze(-1)], dim=-1)          
            disp3 = torch.concat([torch.matmul(frame3.permute(0,2,1), disp3[:,:2].unsqueeze(-1)).squeeze(-1),disp3[:,-1].unsqueeze(-1)], dim=-1)   
            disp4 = torch.concat([torch.matmul(frame4.permute(0,2,1), disp4[:,:2].unsqueeze(-1)).squeeze(-1),disp4[:,-1].unsqueeze(-1)], dim=-1)   
        
        disp = (disp1 + disp2 + disp3 + disp4)/4
        cov = (cov1 + cov2 + cov3 + cov4)/4
        return frame1, disp, cov #, disp_inv, v1,v2

if __name__ == "__main__": ## to quickly check
    input = torch.randn((4,6,200))
    network =  Eq_Motion_Model(tlio_in_dim=6, tlio_out_dim=3,tlio_depths=[2,2,2,2], 
                               tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=2)
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network loaded to device ')
    print("Total number of parameters:", total_params)
    # frame1, pred, cov = network(input)

    ## full cov
    # network =  Eq_Motion_Model(tlio_in_dim=6, tlio_out_dim=3,tlio_depths=[2,2,2,2],
    #                            tlio_net_config_in_dim = 200//32 +1,tlio_cov_dim_out=3)
    # total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    # print('Network loaded to device ')
    # print("Total number of parameters:", total_params)
    # frame1, pred, cov = network(input)

    yaw = torch.randn((1))
    R_o = torch.Tensor([[torch.cos(yaw), torch.sin(yaw), 0], [-torch.sin(yaw), torch.cos(yaw), 0], [0,0,1]])
    v1,v2,frame1, disp, cov = network(input)

    
    
    # rot_out_d = torch.matmul(R_o.unsqueeze(0),out_d.unsqueeze(-1)).squeeze(-1)

    # assert torch.allclose(rot_out_d, out_rot_d, atol = 1e-4), 'vector is not equivariant'
    # assert torch.allclose(out_sca, out_rot_sca, atol= 1e-4), 'covariance is not invariant'

 
   