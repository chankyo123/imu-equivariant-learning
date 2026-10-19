from network.model_resnet import BasicBlock1D, ResNet1D
from network.model_resnet_seq import ResNetSeq1D
from network.model_tcn import TlioTcn

from utils.logging import logging


from network.model_eq_VNCnn_nn_resnet import Eq_Motion_Model as eq_vn_resnet
from network.model_eq_VNCnn_nn_resnet import Eq_Motion_Model_flatten as eq_vn_resnet_flatten
from network.model_eq_VNCnn_nn_resnet import Eq_Motion_Model_flatten_LN as eq_vn_resnet_flatten_ln
from network.rnin_model_lstm import ResNetLSTMSeqNet
from network.model_eq_VNCnn_nn_cpad import Eq_Motion_Model as eq_vn_cnn_cpad
from network.model_eq_VNCnn_nn_cpad import Eq_Motion_Model_flatten as eq_vn_vnn_cpad_flatten
from network.model_resnet_frame import Eq_Motion_Model as resnet_frame_model
from network.model_resnet_frame_pca import Eq_Motion_Model as resnet_frame_pca
from network.model_resnet_frame_frame_avg import Eq_Motion_Model as resnet_frame_avg
from network.model_eq_VNCnn_frame_TLIO_v2_2vectors import Eq_Motion_Model as eq_frame_so2_2scalars_2vec
from network.model_eq_VNCnn_frame_TLIO_v2_2vectors import Eq_Motion_Model_fullCov as eq_frame_so2_fullcov_2vec
from network.model_eq_VNCnn_frame_TLIO_v2_2vectors import Eq_Motion_Model_PearsonCov as eq_frame_so2_pc_2vec
from network.model_eqvncnn_frame_v2_o2 import Eq_Motion_Model as eq_o2_2s
from network.model_eqvncnn_frame_v2_o2 import Eq_Motion_Model_fullCov as eq_o2_fc
from network.model_eqvncnn_frame_v2_o2 import Eq_Motion_Model_PearsonCov as eq_o2_pc
from network.model_eqvncnn_frame_v2_o2 import Eq_Motion_Model_SO3_fullCov as eq_so3_fc
from network.ln_conv_2regress_nochannelmix_bias_slope1 import LN_BasicBlock1D, SO3EquivariantReluBracketLayers    #6D or 9D
import yaml

def get_model(arch, net_config, input_dim=6, output_dim=3):
    if arch == 'eq_o2_frame_2scalars_2vec_deep':
        network = eq_o2_2s(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2,
                               hidden_dim=64, scalar_hidden_dim=64, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_fullCov_2vec_deep':
        network = eq_o2_fc(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=3,
                               hidden_dim=64, scalar_hidden_dim=64, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_PearsonCov_2vec_deep':
        network = eq_o2_pc(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=6,
                               hidden_dim=64, scalar_hidden_dim=64, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_2scalars_2vec_2deep':
        network = eq_o2_2s(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2,
                               hidden_dim=64, scalar_hidden_dim=64, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_fullCov_2vec_2deep':
        network = eq_o2_fc(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=3,
                               hidden_dim=64, scalar_hidden_dim=64, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_so3_frame_fullCov_2vec_2deep':
        # SO(3)-equivariant counterpart of eq_o2_frame_fullCov_2vec_2deep.
        # Input convention: vector (B, T, 3, dim_in=3) -- 3 features (e.g. accel, v1, v2) as 3D vectors
        #                   scalar (B, T, scalar_dim_in=6) -- norms+dot products (rotation invariant)
        # Backbone input channels = dim_in * 3 = 9 (after F.T canonicalization + flatten).
        network = eq_so3_fc(dim_in=3, dim_out=2, scalar_dim_in=6, pooling_dim=1,
                               tlio_in_dim=9, tlio_out_dim=3, tlio_depths=[2,2,2,2],
                               tlio_net_config_in_dim=net_config["in_dim"], tlio_cov_dim_out=3,
                               hidden_dim=64, scalar_hidden_dim=64, depth=2, stride=1,
                               padding='same', kernel=(32,1), bias=False)
    elif arch == 'eq_o2_frame_PearsonCov_2vec_2deep':
        network = eq_o2_pc(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=6,
                               hidden_dim=64, scalar_hidden_dim=64, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_2scalars_2vec':
        network = eq_o2_2s(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2,
                               hidden_dim=128, scalar_hidden_dim=128, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_fullCov_2vec':
        network = eq_o2_fc(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=3,
                               hidden_dim=128, scalar_hidden_dim=128, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_o2_frame_PearsonCov_2vec':
        network = eq_o2_pc(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=6,
                               hidden_dim=128, scalar_hidden_dim=128, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_vn_cnn_wo_t_tlio_frame_2scalars_v2_2vec':
        network = eq_frame_so2_2scalars_2vec(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2,
                               hidden_dim=128, scalar_hidden_dim=128, depth=1, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'eq_vn_cnn_wo_t_tlio_frame_PearsonCov_2vec':
        network = eq_frame_so2_pc_2vec(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=6,
                               hidden_dim=64, scalar_hidden_dim=64, depth=1, stride = 1, padding='same', kernel =(32,1), bias=False
                            )

    elif arch == 'eq_vn_cnn_wo_t_tlio_frame_fullCov_frameop_2vec':
        network = eq_frame_so2_fullcov_2vec(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
                               tlio_in_dim=6, tlio_out_dim=3, tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=3,
                               hidden_dim=128, scalar_hidden_dim=128, depth=2, stride = 1, padding='same', kernel =(32,1), bias=False
                            )

    elif arch =="resnet_2scalars_tlio_frame":
        network = resnet_frame_model(frame_dim_in=6,dim_out=2,frame_hidden_dim=256 , depth=2,tlio_in_dim=6, pooling_dim=-2,
                               tlio_out_dim=3,tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2,
                               stride = 1, padding='same', kernel=16,bias = False)
    elif arch == 'resnet_fullCov_tlio_frame':
        network = resnet_frame_model(frame_dim_in=6,dim_out=2,frame_hidden_dim=256 , depth=2,tlio_in_dim=6, pooling_dim=-2,
                               tlio_out_dim=3,tlio_depths=[2,2,2,2], tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=3,
                               stride = 1, padding='same', kernel=16,bias = False)
    elif arch =="resnet_2scalars_tlio_frame_pca":
        network = resnet_frame_pca(tlio_in_dim=6, tlio_out_dim=3,tlio_depths=[2,2,2,2], 
                                   tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2)
    elif arch =="resnet_2scalars_tlio_frame_frame_avg":
        network = resnet_frame_avg(tlio_in_dim=6, tlio_out_dim=3,tlio_depths=[2,2,2,2], 
                                   tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=2)
    elif arch == 'resnet_fullCov_tlio_frame_pca':
        network = resnet_frame_pca(tlio_in_dim=6,tlio_out_dim=3,tlio_depths=[2,2,2,2],
                                   tlio_net_config_in_dim = net_config["in_dim"],tlio_cov_dim_out=3)
    elif arch == "resnet_w_t": # 9 time positional encoding
        network = ResNet1D(
            BasicBlock1D, input_dim+9, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == 'rnin_vio_model_lstm':
        print('This is RNIN-VIO model!')
        with open('/home/royinakj/rnin-vio/TLIO_output_seqlen1/all_params.yaml', 'r') as f:
            rnin_cfg = yaml.load(f, Loader=yaml.Loader)
        network = ResNetLSTMSeqNet(rnin_cfg)

    elif arch == 'eq_vn_resnet_6v_6s':
        print('Entered eq_vn_resnet_6v_6s for model eq_vn_resnet!')
        network = eq_vn_resnet(dim_in=2, dim_out= 6, scalar_dim_in=14, scalar_dim_out= 6, pooling_dim = 200,depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,1,1], bias=False
                            )
    elif arch == 'eq_vn_resnet_3scalars':
        print('Entered eq_vn_resnet_3scalars for model eq_vn_resnet!')
        network = eq_vn_resnet(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 200,depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,1,1], bias=False
                            )
    elif arch == 'eq_vn_resnet_flatten_6v_6s':
        print('Entered eq_vn_resnet_flatten_6v_6s for model eq_vn_resnet!')
        network = eq_vn_resnet_flatten(dim_in=2, dim_out= 6, scalar_dim_in=14, scalar_dim_out= 6, pooling_dim = 200,depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,2,2], bias=False
                            )
    elif arch == 'eq_vn_resnet_flatten_3scalars':
        print('Entered eq_vn_resnet_flatten_3scalars for model eq_vn_resnet!')
        network = eq_vn_resnet_flatten(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 200,depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,2,2], bias=False
                            )
    elif arch == 'eq_vn_resnet_flatten_ln_3scalars':
        print('Entered eq_vn_resnet_flatten_ln_3scalars for model eq_vn_resnet!')
        network = eq_vn_resnet_flatten_ln(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 200,depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,2,2], bias=False
                            )

    elif arch=='eq_vn_cnn_cpad_6v_6s':
        print('entered eq_vn_cnn_cpad_6v_6s')
        network = eq_vn_cnn_cpad(dim_in=2, dim_out= 6, scalar_dim_in=14, scalar_dim_out= 6, pooling_dim = 1, 
                    hidden_dim=128, scalar_hidden_dim=128, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False)
    elif arch=='eq_vn_cnn_cpad_3scalars':
        print('entered eq_vn_cnn_cpad_3scalars')
        network = eq_vn_cnn_cpad(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 1, 
                    hidden_dim=128, scalar_hidden_dim=128, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False)
    elif arch=='eq_vn_cnn_cpad_wo_t_3scalars':
        print('entered eq_vn_cnn_cpad_wo_t_3scalars')
        network = eq_vn_cnn_cpad(dim_in=2, dim_out= 1, scalar_dim_in=5, scalar_dim_out= 3, pooling_dim = 1, 
                    hidden_dim=128, scalar_hidden_dim=128, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False)
    elif arch == 'eq_vn_resnet_flatten_ln_wo_t_3scalars':
        print('Entered eq_vn_resnet_flatten_ln_wo_t_3scalars for model eq_vn_resnet!')
        network = eq_vn_resnet_flatten_ln(dim_in=2, dim_out= 1, scalar_dim_in=5, scalar_dim_out= 3, pooling_dim = 200,depth=4, 
                               stride = 2, padding=1, kernel =(7,1), grp_sizes = [2,2,2,2], bias=False
                            )
    elif arch=='eq_vn_cnn_cpad_flatten_3scalars':
        print('entered eq_vn_cnn_cpad_flatten_3scalars')
        network = eq_vn_vnn_cpad_flatten(dim_in=2, dim_out= 1, scalar_dim_in=14, scalar_dim_out= 3, pooling_dim = 200, 
                               hidden_dim=128, scalar_hidden_dim=128, depth=3, stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == "resnet_bigger":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [3, 3, 2, 3], net_config["in_dim"]
        )
    # elif arch == "resnet_bigger":
    #     network = ResNet1D(
    #         BasicBlock1D, input_dim, output_dim, [3, 3, 3, 3], net_config["in_dim"]
    #     )
    elif arch == "resnet":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
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
    elif arch == "ln_resnet":
        network = SO3EquivariantReluBracketLayers(
            LN_BasicBlock1D,
            input_dim, output_dim,
            [2, 2, 2, 2],
            # [3, 3, 3, 3],
            # [4, 4, 4, 4],
            net_config["in_dim"],
            # learn_alpha_beta=False,
            # # init_alpha=0.1, init_beta=1.0
            # init_alpha=5.0, init_beta=10.0
        )
    elif arch == "ln_resnet_chol":
        # Ported from imu-equivariant-learning: Equivariant Cholesky-style covariance head.
        # Output: (mean (B,3), A (B,3,3)) with Sigma = A A^T + eps*I.
        from network.ln_conv_2regress_nochannelmix_slope_1_pca_cov import (
            LN_BasicBlock1D_cov,
            SO3EquivariantReluBracketLayers_chol,
        )
        network = SO3EquivariantReluBracketLayers_chol(
            LN_BasicBlock1D_cov,
            input_dim, output_dim,
            [2, 2, 2, 2],
            net_config["in_dim"],
            init_log_a_scale=net_config.get("init_log_a_scale", None),
        )
    else:
        raise ValueError("Invalid architecture: ", arch)

    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    logging.info(f"Number of params for {arch} model is {num_params}")   
    # print(network)

    return network
