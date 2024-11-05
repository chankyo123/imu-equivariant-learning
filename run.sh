#! /bin/bash

echo "Running the script"


venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_mlp/RIDI_resnet18_processeddata/ --epochs 500 --arch resnet18 --input_dim 6 --body_frame True
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet18_processeddata/checkpoints/checkpoint_299.pt --out_dir batch_test_outputs/RIDI_resnet18_new --arch resnet18 --body_frame True --input_dim 6














# windows 1
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --out_dir models_mlp/resnet/ --epochs 300 --arch resnet --input_dim 6 --body_frame True 
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --out_dir models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/ --epochs 300 --arch ln_resnet --input_dim 6 --body_frame True 

# windows 2
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=2 --out_dir models_mlp/resnet_windows2/ --epochs 300 --arch resnet --input_dim 6 --body_frame True 
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=2 --out_dir models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300_windows2/ --epochs 300 --arch ln_resnet --input_dim 6 --body_frame True 

# # continue from pretrain
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=1 --out_dir models_mlp/resnet/frompretrain/ --epochs 400 --arch resnet --input_dim 6 --body_frame True --continue_from pretrain/resnet/checkpoint_best.pt
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=1 --out_dir models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/frompretrain/ --epochs 500 --arch ln_resnet --input_dim 6 --body_frame True --continue_from pretrain/ln_resnet/checkpoint_best.pt


# windows 1
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_mlp/resnet/checkpoint_best.pt --out_dir batch_test_outputs/resnet --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/checkpoints/checkpoint_best.pt --out_dir batch_test_outputs/ln_conv_last_align_notcomp_2regress_noln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_nobias_ep300 --arch ln_resnet --body_frame True --input_dim 6

# windows 2
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --window_time=2 --model_path models_mlp/resnet_windows2/checkpoint_best.pt --out_dir batch_test_outputs/resnet_windows2 --arch resnet --body_frame True --input_dim 6

# pretrain
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_mlp/resnet/frompretrain/checkpoint_best.pt --out_dir batch_test_outputs/resnet/frompretrain --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/frompretrain/checkpoint_best.pt --out_dir batch_test_outputs/ln_conv_last_align_notcomp_2regress_noln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_nobias_ep300/frompretrain --arch ln_resnet --body_frame True --input_dim 6

## RIDI directly test
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path pretrain/resnet/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_direct_test_resnet --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path pretrain/ln_resnet/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_direct_test_ln_resnet --arch ln_resnet --body_frame True --input_dim 6

## RIDI training newstart
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_mlp/RIDI_resnet/ --epochs 300 --arch resnet18 --input_dim 6 --body_frame True

# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/checkpoints/checkpoint_299.pt --out_dir batch_test_outputs/RIDI_resnet18_new --arch resnet18 --body_frame True --input_dim 6

# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_eq/RIDI_ln_resnet/ --epochs 300 --arch ln_resnet --input_dim 6 --body_frame True
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_mlp/RIDI_resnet_add_regular/ --epochs 300 --arch resnet --input_dim 6 --body_frame True
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_mlp/RIDI_resnet_smaller_net_add_regular/ --epochs 300 --arch resnet --input_dim 6 --body_frame True --lr=5e-5

## RIDI training from pretrain
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_mlp/RIDI_resnet/frompretrain/ --epochs 400 --arch resnet --input_dim 6 --body_frame True --continue_from pretrain/resnet/checkpoint_best.pt
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/RIDI/ --out_dir models_eq/RIDI_ln_resnet/frompretrain/ --epochs 500 --arch ln_resnet --input_dim 6 --body_frame True --continue_from pretrain/ln_resnet/checkpoint_best.pt

## RIDI testing
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_resnet --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_eq/RIDI_ln_resnet/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_ln_resnet --arch ln_resnet --body_frame True --input_dim 6

# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/frompretrain/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_resnet/frompretrain --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_eq/RIDI_ln_resnet/frompretrain/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_ln_resnet/frompretrain --arch ln_resnet --body_frame True --input_dim 6

# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/checkpoints/checkpoint_299.pt --out_dir batch_test_outputs/RIDI_resnet_299 --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/checkpoints/checkpoint_149.pt --out_dir batch_test_outputs/RIDI_resnet_149 --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/checkpoints/checkpoint_49.pt --out_dir batch_test_outputs/RIDI_resnet_49 --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet/checkpoints/checkpoint_90.pt --out_dir batch_test_outputs/RIDI_resnet_90 --arch resnet --body_frame True --input_dim 6

# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_eq/RIDI_ln_resnet/checkpoints/checkpoint_299.pt --out_dir batch_test_outputs/RIDI_ln_resnet_299 --arch ln_resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_eq/RIDI_ln_resnet/checkpoints/checkpoint_149.pt --out_dir batch_test_outputs/RIDI_ln_resnet_149 --arch ln_resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_eq/RIDI_ln_resnet/checkpoints/checkpoint_49.pt --out_dir batch_test_outputs/RIDI_ln_resnet_49 --arch ln_resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_eq/RIDI_ln_resnet/checkpoints/checkpoint_90.pt --out_dir batch_test_outputs/RIDI_ln_resnet_90 --arch ln_resnet --body_frame True --input_dim 6

# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet_add_regular/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_resnet_add_regular --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet_add_regular/checkpoints/checkpoint_149.pt --out_dir batch_test_outputs/RIDI_resnet_add_regular_149 --arch resnet --body_frame True --input_dim 6

# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/RIDI/ --model_path models_mlp/RIDI_resnet_smaller_net_add_regular/checkpoint_best.pt --out_dir batch_test_outputs/RIDI_resnet_smaller_net_add_regular --arch resnet --body_frame True --input_dim 6

## RONIN training
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Ronin/ --out_dir models_mlp/RONIN_resnet/ --epochs 300 --arch resnet --input_dim 6 --body_frame True
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Ronin/ --out_dir models_eq/RONIN_ln_resnet/ --epochs 300 --arch ln_resnet --input_dim 6 --body_frame True

## RONIN testing
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Ronin/ --model_path models_mlp/RONIN_resnet/checkpoints/checkpoint_299.pt --out_dir batch_test_outputs/RONIN_resnet --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Ronin/ --model_path models_eq/RONIN_ln_resnet/checkpoint_best.pt --out_dir batch_test_outputs/RONIN_ln_resnet --arch ln_resnet --body_frame True --input_dim 6