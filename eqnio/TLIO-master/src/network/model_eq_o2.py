import torch 

a = torch.randn((1023,3))+0.1
w = torch.randn((1023,3))+0.1

yaw = torch.randn((1))
R_o = torch.Tensor([[-torch.sin(yaw),torch.cos(yaw), 0], [torch.cos(yaw),torch.sin(yaw), 0], [0,0,1]])

w_a = torch.linalg.cross(w,a)
w_w_a = torch.linalg.cross(w,torch.linalg.cross(w,a))

r_a = a@R_o.T
r_w = -w@R_o.T
r_w_r_a = torch.linalg.cross(r_w,r_a)

r_w_a = w_a@R_o.T
assert torch.allclose(r_w_a, r_w_r_a, atol = 1e-6), 'vector is not equivariant'

r_w_w_a = w_w_a@R_o.T
r_w_r_w_a = torch.linalg.cross(r_w,r_w_r_a)
assert torch.allclose(r_w_w_a, r_w_r_w_a, atol = 1e-6), 'vector is not equivariant'

## wz = 0
w2 = w.clone()
w2[:,-1] = 0
x = torch.Tensor([0,0,1]).reshape((1,-1)).repeat(w2.shape[0],1)
r_x = x@R_o.T
r_w2 = -w2@R_o.T
r_x_w = torch.linalg.cross(x,w2)@R_o.T
r_x_r_w = torch.linalg.cross(r_x,r_w2)
assert torch.allclose(r_x_w, r_x_r_w, atol = 1e-6), 'vector is not equivariant'

##case4
R = torch.Tensor([[0,-1,0],[1,0,0],[0,0,1]])
x = w2@R.T
r_w_x = torch.linalg.cross(w,x)@R_o.T

r_ww2 = r_w.clone()
r_ww2[:,-1] = 0
r_x = r_ww2@R.T
r_w_r_x = torch.linalg.cross(r_w,r_x)

assert torch.allclose(r_w_x, r_w_r_x, atol = 1e-6), 'vector is not equivariant'

##case 3
wz = torch.zeros_like(w)
wz[:,-1] = w[:,-1]
x = torch.Tensor([0,0,1]).reshape((1,-1)).repeat(w2.shape[0],1)
r_x_wz = -torch.linalg.cross(x,wz)

r_w = -w@R_o.T
r_wz = torch.zeros_like(r_w)
r_wz[:,-1] = r_w[:,-1]
r_x_r_wz = torch.linalg.cross(x,r_wz)
assert torch.allclose(r_w_x, r_w_r_x, atol = 1e-6), 'vector is not equivariant'


