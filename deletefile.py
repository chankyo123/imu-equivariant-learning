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
directory_path = '/workspace/TLIO/batch_filter_outputs_idso2_uf20/idso2_eq_2res/'
# List of files to keep in each subdirectory
files_to_keep = ['debug.txt', 'not_vio_state.txt.npy', 'metrics.json', 'parameters.json']

# Delete all files except those specified in 'files_to_keep'
delete_files_except(directory_path, files_to_keep)
