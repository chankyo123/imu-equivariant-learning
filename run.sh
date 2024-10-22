#! /bin/bash

echo "Running the script"

# windows 1
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --out_dir models_mlp/resnet/ --epochs 300 --arch resnet --input_dim 6 --body_frame True 
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --out_dir models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/ --epochs 300 --arch ln_resnet --input_dim 6 --body_frame True 

# windows 2
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=2 --out_dir models_mlp/resnet_windows2/ --epochs 300 --arch resnet --input_dim 6 --body_frame True 
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=2 --out_dir models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300_windows2/ --epochs 300 --arch ln_resnet --input_dim 6 --body_frame True 

# # # continue from pretrain
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=1 --out_dir models_mlp/resnet/frompretrain/ --epochs 400 --arch resnet --input_dim 6 --body_frame True --continue_from pretrain/resnet/checkpoint_best.pt
# venv/bin/python3 src/main_net.py --mode train --root_dir local_data_bodyframe/Euroc/ --window_time=1 --out_dir models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/frompretrain/ --epochs 500 --arch ln_resnet --input_dim 6 --body_frame True --continue_from pretrain/ln_resnet/checkpoint_best.pt


# windows 1
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_mlp/resnet/checkpoints/checkpoint_69.pt --out_dir batch_test_outputs/resnet --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/checkpoints/checkpoint_69.pt --out_dir batch_test_outputs/ln_conv_last_align_notcomp_2regress_noln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_nobias_ep300 --arch ln_resnet --body_frame True --input_dim 6

# windows 2
venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --window_time=2 --model_path models_mlp/resnet/checkpoints/checkpoint_69.pt --out_dir batch_test_outputs/resnet --arch resnet --body_frame True --input_dim 6


# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_mlp/resnet/frompretrain/checkpoint_best.pt --out_dir batch_test_outputs/resnet/frompretrain --arch resnet --body_frame True --input_dim 6
# venv/bin/python3 src/main_net.py --mode test --root_dir local_data_bodyframe/Euroc/ --model_path models_eq/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_largechannel_onechannelmix_mselss_nobias_ep300/frompretrain/checkpoint_best.pt --out_dir batch_test_outputs/ln_conv_last_align_notcomp_2regress_noln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_nobias_ep300/frompretrain --arch ln_resnet --body_frame True --input_dim 6

# pretrain