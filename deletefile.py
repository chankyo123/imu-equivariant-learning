import os
import shutil

def delete_files_except(directory, exceptions):
    for root, dirs, files in os.walk(directory):
        print(root)
        for file in files:
            file_path = os.path.join(root, file)
            if file not in exceptions:
                print(file_path)
                os.remove(file_path)

# Replace 'your_directory' with the path to your target directory
# directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_uf10/resnet/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_uf10/res_2res_nodrop/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_idso2_uf10/idso2_resnet_2/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_idso3_uf10/idso3_eq_2res_200hz_3input_mselss_0.5_pertb/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_idso3_notfixed_uf10/notfixed_idso3_eq_2res_200hz_3input_mselss_0.5_pertb/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_uf10/june_sim_resnet/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_idso2_uf10/test_idso2_eq_2res_200hz_3input_mselss_0.5_pertb/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_idso3_2_uf200/eq_2res_200hz_2input_mselss/'
directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_uf200/gtv_uf200/eq_2res_20hz_3input_samelss_nobias_ep200/'

# directory_path = '/workspace/imu-equivariant-learning/batch_filter_outputs_sim_notjune_uf10/sim_eq_2res_200hz_3input_mselss/'

# List of files to keep in each subdirectory
files_to_keep = ['debug.txt', 'not_vio_state.txt.npy', 'metrics.json', 'parameters.json']

# Delete all files except those specified in 'files_to_keep'
delete_files_except(directory_path, files_to_keep)
