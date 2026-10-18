# from network.eq_resnet_1res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_2res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_3res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_4res_bd import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D

#### regress 2 output (mean, covariance)
# from network.eq_resnet_2res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_4res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.eq_resnet_4res_2regress_more_param import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.vn_resnet_4res_2regress import VN_BasicBlock1D, VN_ResNet1D    #6D or 9D
# from network.ln_2regress import SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress import SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_less_param import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix_slope_1_largechannel_onechannelmix import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D

from network.ln_4res_peract import LN_PreActBracketBlock1D, SO3EquivariantReluBracketLayers_peract    #6D or 9D
# from network.ln_conv_4regress_nochannelmix_bias_dp import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
from network.ln_conv_2regress_nochannelmix_bias import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix_slope_1 import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix_slope_1_bias import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
from network.ln_conv_2regress_nochannelmix_slope_1_pca_cov import (
    LN_BasicBlock1D_cov,
    SO3EquivariantReluBracketLayers_cov,
    SO3EquivariantReluBracketLayers_chol,   # ← 새로 추가
)

# from network.ln_conv_2regress_nochannelmix_slope_1_onechannelmix import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
# from network.ln_conv_2regress_nochannelmix_slope_1_onechannelmix_param16_fc2 import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D

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
# from network.model_resnet_2res_nodrop import BasicBlock1D, ResNet1D


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
            # VN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
            VN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"], True
        )
    elif arch == "resnet_seq":
        network = ResNetSeq1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "ln_resnet":
            network = SO3EquivariantReluBracketLayers(
                LN_BasicBlock1D,
                input_dim, output_dim, 
                [2, 2, 2, 2],
                net_config["in_dim"]
            )
    elif arch == "ln_resnet_peract":
            network = SO3EquivariantReluBracketLayers_peract(
                LN_PreActBracketBlock1D,   # 사실 내부에서 새 block을 강제로 사용함
                input_dim,
                output_dim,
                [3, 4, 6, 3],              # deeper within 4 stages
                net_config["in_dim"],
                stem_width=24,
                widths=(48, 96, 192, 384),
                head_hidden=384,
                max_drop_path=0.10,
                feature_drop=0.10,
                gate_reduction=4,
            )
    elif arch == "ln_resnet_cov":
            network = SO3EquivariantReluBracketLayers_cov(
                LN_BasicBlock1D_cov,
                input_dim, output_dim, 
                [2, 2, 2, 2],
                net_config["in_dim"]
            )
    elif arch == "ln_resnet_chol":
        # Equivariant Cholesky-style covariance head (A only; Sigma = A A^T + eps*I)
        network = SO3EquivariantReluBracketLayers_chol(
            LN_BasicBlock1D_cov,
            input_dim, output_dim,
            [2, 2, 2, 2],
            net_config["in_dim"],
            init_log_a_scale=net_config.get("init_log_a_scale", None),
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
