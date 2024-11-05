# from network.eq_resnet_1res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_2res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_3res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_4res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D

#### regress 2 output (mean, covariance)
from network.eq_resnet_4res_2regress_cl import VN_BasicBlock1D_cl, VN_ResNet1D_cl    #6D or 9D

# from network.eq_resnet_2res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D

from network.eq_resnet_4res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_4res_2regress_more_param import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.vn_resnet_1conv_4res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.vn_resnet_4res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D

# from network.ln_conv_2regress_less_param import VN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_killing import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_onechannelmix_slope_2 import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix_slope_1_largechannel import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
from network.ln_conv_2regress_nochannelmix_slope_1 import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix_slope_1_onechannelmix import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_onechannelmix import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D

# from network.eq_resnet_4res_2regress_bias import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D

# from network.eq_resnet_2res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_1res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_3res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_4res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_linear_2res import VN_BasicBlock1D, VN_ResNet1D
# from network.eq_resnet_linear_1res import VN_BasicBlock1D, VN_ResNet1D

# from network.model_resnet_1res_nodrop import BasicBlock1D, ResNet1D
# from network.model_resnet_2res_nodrop import BasicBlock1D, ResNet1D
# from network.model_resnet_3res_nodrop import BasicBlock1D, ResNet1D
# from network.model_resnet_4res_nodrop import BasicBlock1D, ResNet1D

# from network.model_vn_resnet_2 import VN_BasicBlock1D, VN_ResNet1D

from network.model_resnet import BasicBlock1D, ResNet1D
# from network.model_resnet_1res_nodrop import BasicBlock1D, ResNet1D


# from network.model_resnet_2res_nodrop import BasicBlock1D, ResNet1D

# from network.model_resnet_seq import ResNetSeq1D
# from network.model_tcn import TlioTcn

from utils.logging import logging


def get_model(arch, net_config, input_dim=6, output_dim=3, close_loop = False):
    if arch == "resnet":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "resnet18":
        _input_channel, _output_channel = 6, 3
        _fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
        from network.resnet18 import ResNet1D as ResNet1D18
        from network.resnet18 import FCOutputModule as FCOutputModule18
        from network.resnet18 import BasicBlock1D as BasicBlock1D18
        network = ResNet1D18(_input_channel, _output_channel, BasicBlock1D18, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule18, kernel_size=3, **_fc_config)
    elif arch == "vn_resnet":
        if close_loop:
            network = VN_ResNet1D_cl(
                # VN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
                VN_BasicBlock1D_cl, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"], True
            )
        else:
            network = VN_ResNet1D(
                # VN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
                VN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"], True
            )
    elif arch == "ln_resnet":
            network = SO3EquivariantReluBracketLayers(
                LN_BasicBlock1D,
                input_dim, output_dim, 
                [2, 2, 2, 2],
                net_config["in_dim"]
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
