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

# json_file = "../TLIO/batch_filter_outputs_uf20/models-resnet/metrics.json"  #1.7251 / 0.1260-0.7001-2.1484
# json_file = "/batch_filter_outputs_uf10/resnet/metrics.json"  #1.7251 / 0.1260-0.7001-2.1484
# json_file = "./batch_filter_outputs_uf10/models-resnet/metrics.json"  #1.70722854503/ 0.1278297560347-0.7001-2.1484

# json_file = "./batch_filter_outputs_uf10/eq_2res_mselss_uf10/eq_2res_200hz_3input_mselss/metrics.json"  #3.3359160136934287 / 0.20138467
# json_file = "./batch_filter_outputs_uf10/eq_2res_mselss_nobias_est_uf10/eq_2res_200hz_3input_mselss/metrics.json"  #3.30234139322 / 0.1664199
# json_file = "./batch_filter_outputs_uf10/eq_2res_mselss_0.5_pertb_epmore2_N_1_uf10/eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"  #5.1379399 / 0.173423197
# json_file = "./batch_filter_outputs_uf10/eq_2res_mselss_0.5_pertb_N_1_uf10/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"  #3.1010206447 / 0.1599918932  << -- 이게 젤 좋은 듯
# json_file = "./batch_filter_outputs_uf10/eq_2res_mselss_0.5_pertb_epmore2_N_1_uf10/models-eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"  #5.137939965969 / 0.173423

# json_file = "batch_test_outputs/resnet/metrics.json"   #1.84603557 / 2.2383225
# json_file = "../TLIO/batch_test_outputs/resnet/metrics.json"   #1.817235 / 2.18

# json_file = "batch_test_outputs/res_2res_nodrop/metrics.json"   #2.233789 / 2.6531212284

# json_file = "batch_test_outputs/sim_body_previous_idso2/metrics.json"   #10.01756400 / 11.385903673970976
# json_file = "batch_test_outputs/delete_idso2_fixed2/metrics.json"   #9.89564424 / 11.2994975213

# json_file = "batch_test_outputs/idso2_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #7.022521 / 7.75489
# json_file = "batch_test_outputs/idso2_csv_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #7.36309 / 8.0417
# json_file = "batch_test_outputs/idso2_csv_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #7.36309 / 8.0417
# json_file = "batch_test_outputs/idso3_resnet/metrics.json"   #96.003399 / 110.221952
# json_file = "batch_test_outputs/idso2_resnet/metrics.json"   #1.7290 / 2.088958
# json_file = "batch_test_outputs/idso3_resnet/metrics.json"   #96.003399 / 110.221952

# json_file = "batch_test_outputs/idso2_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #4.78175 / 5.1652 / 0.15657479
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_ep123/metrics.json"   #8.9868 / 9.692717 / 0.32903
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_0.5pertb_ep165/metrics.json"   #16.202142 / 8.029240151 / 0.34986
# json_file = "batch_test_outputs/idso2_csv_eq_2res_200hz_4input_ep99/metrics.json"   #10.2874 / 11.312
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_ep99/metrics.json"   #9.08757147 / 9.816118
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_ep80/metrics.json"   #9.58840560 / 10.342239317

# json_file = "batch_test_outputs/idso3_csv_eq_2res_200hz_4input_ep99/metrics.json"   #24.7229437 / 27.79573660
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_4input_ep99/metrics.json"   #23.67606 / 26.18890076
json_file = "batch_test_outputs/idso3_eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"   #6.674809036863 / 7.17102359306 / 0.25051443
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_4input_ep123/metrics.json"   #15.306714955 / 16.86535730 / 0.52134067772
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_4input_0.5pertb_ep165/metrics.json"   #20.58666 / 22.9058 / 0.4827


# json_file = "batch_test_outputs/eq_2res_200hz_4input_ep80/metrics.json"   #3.54345552609 / 4.017126 / 0.142786
# json_file = "batch_test_outputs/eq_2res_200hz_4input_ep123/metrics.json"   #3.34875253661426 / 3.789730 / 0.133533958759
# json_file = "batch_test_outputs/eq_2res_200hz_4input_0.5pertb_ep165/metrics.json"   #3.25278 / 3.631963 / 0.11838


# json_file = "batch_test_outputs/eq_2res_200hz_3input/metrics.json"   #3.4788114 / 3.94063
# json_file = "batch_test_outputs/eq_2res_200hz_3input_ep_more2/metrics.json"   #3.52228753 / 3.938439400
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss/metrics.json"   #3.32894499 / 3.71997
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_ep200/metrics.json"   #3.32894499 / 3.71997
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_less_v_pertb/metrics.json"   #3.111072 / 3.425042299
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_less_v_pertb_ep200/metrics.json"   #3.3402726 / 3.65995785
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #3.1371065 / 3.4769123
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_nodetach/metrics.json"   #5.0732262 / 5.71939  
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"   #2.813362 / 3.060839 / 0.068971 <---- 이게 제일 좋은듯
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_no_pertb_atall_ep200/metrics.json"   #2.6725 / 2.886775  
# json_file = "batch_test_outputs/eq_1res_200hz_3input_mselss_no_pertb_atall_ep200/metrics.json"   #2.6276942 /  2.8366642738  

# json_file = "batch_test_outputs/sim_world/metrics.json"   #10.72693 /  12.05368
# json_file = "batch_test_outputs/sim_world_idso2_2_worldframe/metrics.json"   #22.987536403847 /  25.737584164400
# json_file = "batch_test_outputs/sim_world_idso3_2_worldframe/metrics.json"   #354.1185 / 408.01704 /  5.799109333753586

# json_file = "batch_test_outputs/sim_body/metrics.json"   #11.3146 /  12.72
# # json_file = "batch_test_outputs/sim_body_ep99/metrics.json"   #10.77844 /  12.0627677
# json_file = "batch_test_outputs/sim_body_idso2/metrics.json"   #13.4334099 /  15.146993
# json_file = "batch_test_outputs/sim_idso2_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #14.1992214182 /  15.876565 / 0.8945234946906566
# json_file = "batch_test_outputs/sim_idso2_eq_2res_200hz_3input_mselss/metrics.json"   #15.183314 /  16.98737712 / 0.843082719

# json_file = "batch_test_outputs/sim_body_idso3/metrics.json"   #80.197805 /  92.912570675
# json_file = "batch_test_outputs/sim_idso3_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #44.121510 /  50.42309706628    / 1.06438146
# json_file = "batch_test_outputs/sim_idso3_eq_2res_200hz_3input_mselss/metrics.json"   #47.724336 /  54.6730013 / 0.9397254191339016
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #12.6049 / 14.20915


# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss/metrics.json"
# json_file = "batch_test_outputs/eq_2res_200hz_3input/metrics.json"
# json_file = "batch_test_outputs/eq_2res_200hz_3input_ep_more2/metrics.json"
# json_file = "batch_test_outputs/eq_2res_200hz_3input_big_v_pertb/metrics.json"

print('json file : ', json_file)
with open(json_file, "r") as f:
    data = json.load(f)

# Initialize lists to store the ate and rpe_rmse_1000 values
ate_values = []
rmse_values = []
rmse_vel_values = []

for key, value in data.items():
    # Access the "ronin" dictionary
    # filter_dict = value.get("filter")
    filter_dict = value.get("ronin")
    
    if filter_dict:
        ate_value = filter_dict.get("ate")
        # rmse_value = filter_dict.get("rpe_rmse_1000")
        rmse_value = filter_dict.get("rmse")
        rmse_vel_value = filter_dict.get("rmse_vel")
        
        if ate_value is not None:
            ate_values.append(ate_value)
        
        if rmse_value is not None:
            rmse_values.append(rmse_value)
        
        if rmse_vel_value is not None:
            rmse_vel_values.append(rmse_vel_value)
        
        # Print the current ate, rmse, and rmse_vel values
        print(f"Key: {key}, ate value: {ate_value}, rmse value: {rmse_value}, rmse_vel value: {rmse_vel_value}")

# Calculate the average ate and rpe_rmse_1000 (if there are any values)
if ate_values:
    average_ate = sum(ate_values) / len(ate_values)
    print(f"Average ate value in 'filter': {average_ate}")
    print(f"Number of 'ate' values: {len(ate_values)}")
else:
    print("No 'ate' values found in 'filter' key")

if rmse_values:
    average_rmse = sum(rmse_values) / len(rmse_values)
    print(f"Average rmse value in 'filter': {average_rmse}")
    print(f"Number of 'rmse' values: {len(rmse_values)}")
else:
    print("No 'rmse' values found in 'filter' key")
    
if rmse_vel_values:
    average_rmse_vel = sum(rmse_vel_values) / len(rmse_vel_values)
    print(f"Average rmse_vel value in 'filter': {average_rmse_vel}")
    print(f"Number of 'rmse_vel' values: {len(rmse_vel_values)}")
else:
    print("No 'rmse_vel' values found in 'filter' key")
