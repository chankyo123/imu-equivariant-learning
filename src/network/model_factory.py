from network.eq_resnet_2res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_1res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_2res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_3res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_4res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_linear_2res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_linear_1res import VN_BasicBlock1D, VN_ResNet1D

# from network.model_resnet_1res_nodrop import BasicBlock1D, ResNet1D
from network.model_resnet_2res_nodrop import BasicBlock1D, ResNet1D
# from network.model_resnet_3res_nodrop import BasicBlock1D, ResNet1D
# from network.model_resnet_4res_nodrop import BasicBlock1D, ResNet1D

# from network.model_vn_resnet_2 import VN_BasicBlock1D, VN_ResNet1D

# from network.model_resnet import BasicBlock1D, ResNet1D


# from network.model_resnet_2res_nodrop import BasicBlock1D, ResNet1D

# from network.model_resnet_seq import ResNetSeq1D
# from network.model_tcn import TlioTcn

from utils.logging import logging


def get_model(arch, net_config, input_dim=6, output_dim=3):
    if arch == "resnet":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "vn_resnet":
        network = VN_ResNet1D(
            VN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "resnet_seq":
        network = ResNetSeq1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "tcn":
        network = TlioTcn(
            input_dim,
            output_dim,
            [64, 64, 64, 64, 128, 128, 128],
            kernel_size=2,
            dropout=0.2,
            activation="GELU",
        )
    else:
        raise ValueError("Invalid architecture: ", arch)

    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    logging.info(f"Number of params for {arch} model is {num_params}")   

    return network
