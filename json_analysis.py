import json

# Open the JSON file
# json_file = "eq_more-imu0npy_uf10/eq_2res_200hz_3input_ep_more2/metrics.json"
# json_file = "eq_more-imu0npy-N_0.01_uf10/eq_2res_200hz_3input_ep_more2/metrics.json"
# json_file = "batch_filter_outputs_uf20/models-resnet/metrics.json"

json_file = "batch_filter_outputs_uf10/eq_2res_epmore2_N_0.01_uf10/eq_2res_200hz_3input_ep_more2/metrics.json"
old_part = "eq_2res_epmore2_N_0.01_uf10"

# new_part = "eq_2res_epmore2_N_0.01_uf10"  #10.20 / 0.8507674
new_part = "eq_2res_epmore2_N_0.1_uf10"  #3.127847 / 0.19080289-0.95374-3.378675
# new_part = "eq_2res_epmore2_N_1_uf10"   #3.1064 / 0.1584-0.9317-3.28
# new_part = "eq_2res_epmore2_N_3_uf10"   #3.1951 / 0.1537-0.9510-3.35542
# new_part = "eq_2res_epmore2_upd_rmbias_dontupd_yaw_uf10"   #13.081 / 0.969
# new_part = "eq_2res_epmore2_fixpropag_bias_uf10"   #10.20690 / 0.8507
# new_part = "eq_2res_epmore2_fixpropag_bias_uf10"   #10.20690 / 0.8507
# new_part = "eq_2res_epmore2_meas-cov_bd_uf10"   #3.56 / 0.26007
# new_part = "eq_2res_epmore2_meas-cov_wld_uf10"   #4.465811 / 0.4563

json_file = json_file.replace(old_part, new_part)

json_file = "../TLIO/batch_filter_outputs_uf20/models-resnet/metrics.json"  #1.7251 / 0.1260-0.7001-2.1484
# json_file = "/batch_filter_outputs_uf10/resnet/metrics.json"  #1.7251 / 0.1260-0.7001-2.1484

# json_file = "batch_test_outputs/resnet/metrics.json"   #1.84603557 / 2.2383225
# json_file = "../TLIO/batch_test_outputs/resnet/metrics.json"   #1.817235 / 2.18

# json_file = "batch_test_outputs/res_2res_nodrop/metrics.json"   #2.233789 / 2.6531212284

# json_file = "batch_test_outputs/eq_2res_200hz_3input/metrics.json"   #3.4788114 / 3.94063
# json_file = "batch_test_outputs/eq_2res_200hz_3input_ep_more2/metrics.json"   #3.52228753 / 3.938439400
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss/metrics.json"   #3.32894499 / 3.71997
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_ep200/metrics.json"   #3.32894499 / 3.71997
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_less_v_pertb/metrics.json"   #3.111072 / 3.425042299
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_less_v_pertb_ep200/metrics.json"   #3.3402726 / 3.65995785
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #3.1371065 / 3.4769123
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb_ep200/metrics.json"   #2.813362 / 3.060839  <---- 이게 제일 좋은듯
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_nodetach/metrics.json"   #5.0732262 / 5.71939  <---- 이게 제일 좋은듯


# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss/metrics.json"
# json_file = "batch_test_outputs/eq_2res_200hz_3input/metrics.json"
# json_file = "batch_test_outputs/eq_2res_200hz_3input_ep_more2/metrics.json"
# json_file = "batch_test_outputs/eq_2res_200hz_3input_big_v_pertb/metrics.json"

print('json file : ', json_file)
with open(json_file, "r") as f:
    data = json.load(f)

# Initialize lists to store the ate and rpe_rmse_1000 values
ate_values = []
rpe_rmse_1000_values = []

# Loop through each key in the data dictionary
for key, value in data.items():
    # Access the "filter" dictionary
    
    filter_dict = value.get("filter")
    # filter_dict = value.get("ronin")
    
    # Check if "filter" exists and has "ate" and "rpe_rmse_1000" keys
    if filter_dict:
        ate_value = filter_dict.get("ate")
        
        rpe_rmse_1000_value = filter_dict.get("rpe_rmse_1000")
        # rpe_rmse_1000_value = filter_dict.get("rmse")
        
        if ate_value is not None:
            ate_values.append(ate_value)
        
        if rpe_rmse_1000_value is not None:
            rpe_rmse_1000_values.append(rpe_rmse_1000_value)
        
        # Print the current ate and rpe_rmse_1000 values
        print(f"Key: {key}, ate value: {ate_value}, rpe_rmse_1000 value: {rpe_rmse_1000_value}")

# Calculate the average ate and rpe_rmse_1000 (if there are any values)
if ate_values:
    average_ate = sum(ate_values) / len(ate_values)
    print(f"Average ate value in 'filter': {average_ate}")
    print(f"Number of 'ate' values: {len(ate_values)}")
else:
    print("No 'ate' values found in 'filter' key")

if rpe_rmse_1000_values:
    average_rpe_rmse_1000 = sum(rpe_rmse_1000_values) / len(rpe_rmse_1000_values)
    print(f"Average rpe_rmse_1000 value in 'filter': {average_rpe_rmse_1000}")
    print(f"Number of 'rpe_rmse_1000' values: {len(rpe_rmse_1000_values)}")
else:
    print("No 'rpe_rmse_1000' values found in 'filter' key")
