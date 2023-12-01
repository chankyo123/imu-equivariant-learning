# import json

# # Load JSON data from file
# # with open('./batch_test_outputs/models-eq_1res/metrics.json', 'r') as file: 
# with open('./batch_test_outputs/train_test_so3_eq_1res/metrics.json', 'r') as file: 
# # with open('../TLIO/test_outputs/metrics.json', 'r') as file:
# # with open('../TLIO/batch_test_outputs/models-resnet/metrics.json', 'r') as file:
#     data = json.load(file)

# # Extract RMSE values for each sequence
# rmse_values = [sequence_data['ronin']['rmse'] for sequence_data in data.values()]

# # Calculate the mean RMSE
# mean_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else None

# # Print or use the mean value
# print(f"Mean RMSE across all sequences: {mean_rmse}")

import json

# Load JSON data from file
# with open('./batch_filter_outputs_idso2_uf20/models-eq_2res/metrics.json', 'r') as file: 
# with open('../TLIO/test_outputs/metrics.json', 'r') as file:
# with open('./batch_filter_outputs_so2so2_uf20/so2id_res_1res/metrics.json', 'r') as file: 
# with open('./batch_filter_outputs_idso2_uf20/idso2_eq_2res/metrics.json', 'r') as file: 
with open('./batch_filter_outputs_uf20/models-eq_2res/metrics.json', 'r') as file: 
# with open('../TLIO/batch_test_outputs/idso3_eq_2res_2.0/metrics.json', 'r') as file:
    data = json.load(file)

# Extract RMSE values for each sequence
# rmse_values = [sequence_data['ronin']['rmse'] for sequence_data in data.values()]
rmse_values = [sequence_data['filter']['ate'] for sequence_data in data.values()]

# Calculate the mean RMSE
mean_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else None

# Print or use the mean value
print(f"Mean RMSE across all sequences: {mean_rmse}")
