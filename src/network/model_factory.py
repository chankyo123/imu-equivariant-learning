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
from network.ln_conv_2regress_nochannelmix_slope_1 import (
    LN_BasicBlock1D as LN_BasicBlock1D_slope1_nobias,
    SO3EquivariantReluBracketLayers as SO3EquivariantReluBracketLayers_slope1_nobias,
)  # slope-0.1 backbone with bias-free FcBlocks: the architecture of the released RIDI GINIO-W checkpoint
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
from network.model_imunet import IMUNet
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
    elif arch == "imunet":
        # IMUNet baseline (depthwise-separable 1D CNN); (mean, logstd) head -> cov_type='diag' path.
        network = IMUNet(input_dim, output_dim)
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
    elif arch == "ln_resnet_slope1_nobias":
            # ln_resnet with VNLeakyReLU slope 0.1 and bias-free FcBlocks (ln_conv_2regress_nochannelmix_slope_1.py).
            # The released RIDI GINIO-W checkpoint (models_ridi/world_ln_2regress_slope1) strict-loads ONLY into this
            # variant; the TLIO / NanoBench ln_resnet checkpoints use "ln_resnet" (slope 0.0, biased FcBlocks).
            network = SO3EquivariantReluBracketLayers_slope1_nobias(
                LN_BasicBlock1D_slope1_nobias,
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
    elif arch == "ln_resnet_cov_diag":
            # Same backbone/mean as ln_resnet_cov but a coordinate-frame DIAGONAL covariance head.
            # Load a spectral base with --freeze_mean to build the diagonal arm of the matched pair.
            network = SO3EquivariantReluBracketLayers_cov(
                LN_BasicBlock1D_cov,
                input_dim, output_dim,
                [2, 2, 2, 2],
                net_config["in_dim"],
                cov_head='diag'
            )
    # ---- ln189-anchored spectral/diagonal covariance heads (Fig. 5(b) matched pair) ----
    # The ln189 TLIO checkpoint (checkpoints/tlio/ginio_w_ln189) is an `ln_resnet`
    # (ln_conv_2regress_nochannelmix_bias.py, VNLeakyReLU slope 0.0). These archs keep that
    # backbone + mean head (loaded with strict=False, frozen with --freeze_mean) and add a fresh
    # covariance head so that a diagonal-vs-spectral comparison differs ONLY in the covariance
    # parametrization. Use the `_s0` variants: they subclass the deployed slope-0.0 backbone.
    # `ln189_speccov` / `ln189_speccov_diag` subclass the slope-0.1 backbone
    # (ln_conv_2regress_nochannelmix_slope_1_bias.py); the parameter names are identical, so the
    # ln189 checkpoint loads, but the forward pass differs from the deployed model (NN median ATE
    # 1.879 vs 0.906). They are kept only so that older runs can be reproduced.
    elif arch == "ln189_speccov":
            from network.ln_conv_2regress_nochannelmix_bias_speccov import (
                SO3EquivariantReluBracketLayers_speccov)
            from network.ln_conv_2regress_nochannelmix_slope_1_bias import (
                LN_BasicBlock1D as LN_BasicBlock1D_189)
            network = SO3EquivariantReluBracketLayers_speccov(
                LN_BasicBlock1D_189,
                input_dim, output_dim,
                [2, 2, 2, 2],
                net_config["in_dim"],
                cov_head='spectral'
            )
    elif arch == "ln189_speccov_diag":
            # matched-pair DIAGONAL arm on the slope-0.1 backbone (coordinate-frame diagonal head).
            from network.ln_conv_2regress_nochannelmix_bias_speccov import (
                SO3EquivariantReluBracketLayers_speccov)
            from network.ln_conv_2regress_nochannelmix_slope_1_bias import (
                LN_BasicBlock1D as LN_BasicBlock1D_189d)
            network = SO3EquivariantReluBracketLayers_speccov(
                LN_BasicBlock1D_189d,
                input_dim, output_dim,
                [2, 2, 2, 2],
                net_config["in_dim"],
                cov_head='diag'
            )
    elif arch == "ln189_speccov_s0":
            # ln189's DEPLOYED slope-0.0 backbone + mean (ln_conv_2regress_nochannelmix_bias) + fresh
            # spectral cov head. Load the ln189 checkpoint (strict=False) + --freeze_mean -> the mean is
            # forward-identical to the deployed ln189. Trains ONLY the spectral head.
            from network.ln_conv_2regress_nochannelmix_bias_speccov_slope0 import (
                SO3EquivariantReluBracketLayers_speccov_s0)
            network = SO3EquivariantReluBracketLayers_speccov_s0(
                LN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2],
                net_config["in_dim"], cov_head='spectral')
    elif arch == "ln189_speccov_s0_diag":
            # matched-pair DIAGONAL arm on the SAME slope-0.0 backbone / frozen ln189 mean.
            from network.ln_conv_2regress_nochannelmix_bias_speccov_slope0 import (
                SO3EquivariantReluBracketLayers_speccov_s0)
            network = SO3EquivariantReluBracketLayers_speccov_s0(
                LN_BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2],
                net_config["in_dim"], cov_head='diag')
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
