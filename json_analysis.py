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

json_file = "./batch_filter_outputs_uf10/models-resnet/metrics.json"  #1.707228 / 0.127829753  / 156
json_file = "./batch_filter_outputs_uf20/eq_2res_300hz_2input_samelss_yesbias2_ep200/metrics.json"  #1.707228 / 0.127829753  / 156

# json_file = "./batch_filter_outputs_uf10/use_meas_cov_bd_uf10/eq_2res_200hz_3input_samelss_TLIO/metrics.json"  #3.9676 / 0.323602
# json_file = "./batch_filter_outputs_biasupdate_uf10/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"  #4.66627245 / 0.21465 / 
# json_file = "./batch_filter_outputs_nobiasupdate_uf10/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"  #4.686 / 0.21569281

# json_file = "../TLIO/batch_filter_outputs_uf10/eq_2res_mselss_0.5_pertb_uf10/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"  # 2.98538 / 0.18595 / 161.202913
# json_file = "./batch_filter_outputs_idso3_2_uf20/idso3_3input_eq_2res_bd_20hz/metrics.json"  # 24.4039031
# json_file = "./batch_filter_outputs_idso3_2_uf200/idso3_3input_eq_2res_bd_20hz/metrics.json"  # 23.5512

# json_file = "batch_test_outputs/resnet/metrics.json"   #1.84603557 / 2.2383225
# json_file = "batch_test_outputs/resnet_2/metrics.json"   #1.8172 / 2.184373 / 0.1281991
# json_file = "../TLIO/batch_test_outputs/resnet/metrics.json"   #1.817235 / 2.18

# json_file = "batch_test_outputs/res_2res_nodrop/metrics.json"   #2.233789 / 2.6531212284
# json_file = "batch_test_outputs/eq_2res_200hz_3input_ep49/metrics.json"   #4.56303 / 5.121736
# json_file = "batch_test_outputs/eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #3.808783 / 4.310868
# json_file = "batch_test_outputs/eq_2res_200hz_4input/metrics.json"   #3.46013 / 3.9097141
# json_file = "batch_test_outputs/eq_2res_200hz_4input_ep123/metrics.json"   #3.37680 /  3.83612111
# json_file = "batch_test_outputs/eq_2res_200hz_4input_ep200/metrics.json"   #3.335885 /  3.78153
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #3.1371065 / 3.4769123

# json_file = "batch_test_outputs/sim_body_previous_idso2/metrics.json"   #10.01756400 / 11.385903673970976
# json_file = "batch_test_outputs/delete_idso2_fixed2/metrics.json"   #9.89564424 / 11.2994975213

# json_file = "batch_test_outputs/idso2_eq_2res_200hz_3input_mselss_0.5_pertb_previous/metrics.json"   #4.11031 / 4.523512 / 0.16568
# json_file = "batch_test_outputs/idso2_resnet/metrics.json"   #1.81273 / 2.19659 / 0.128702
# json_file = "batch_test_outputs/test_idso2_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #4.0559 / 4.4110344 / 0.13914
# json_file = "batch_test_outputs/test_idso2_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #4.0559 / 4.4110344 / 0.13914

# json_file = "batch_test_outputs/idso2_csv_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #7.36309 / 8.0417
# json_file = "batch_test_outputs/idso2_csv_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #7.36309 / 8.0417
# json_file = "batch_test_outputs/idso3_resnet/metrics.json"   #96.003399 / 110.221952
# json_file = "batch_test_outputs/idso3_resnet/metrics.json"   #96.003399 / 110.221952

# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_ep123/metrics.json"   #8.9868 / 9.692717 / 0.32903
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_0.5pertb_ep165/metrics.json"   #16.202142 / 8.029240151 / 0.34986
# json_file = "batch_test_outputs/idso2_csv_eq_2res_200hz_4input_ep99/metrics.json"   #10.2874 / 11.312
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_ep99/metrics.json"   #9.08757147 / 9.816118
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_4input_ep80/metrics.json"   #9.58840560 / 10.342239317
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"   #4.29106390 / 4.650624 / 0.1446
# json_file = "batch_test_outputs/idso2_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   # 4.1103 / 4.52 / 0.16

# json_file = "batch_test_outputs/idso3_csv_eq_2res_200hz_4input_ep99/metrics.json"   #24.7229437 / 27.79573660
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_4input_ep99/metrics.json"   #23.67606 / 26.18890076

# json_file = "batch_test_outputs/idso3_eq_2res_200hz_4input_ep123/metrics.json"   #15.306714955 / 16.86535730 / 0.52134067772
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_4input_0.5pertb_ep165/metrics.json"   #20.58666 / 22.9058 / 0.4827
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"   #5.4796 /  5.928642002 /  0.21871
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #5.8812171 /  6.390807 /  0.2166


# json_file = "batch_test_outputs/eq_2res_200hz_4input_ep80/metrics.json"   #3.54345552609 / 4.017126 / 0.142786
# json_file = "batch_test_outputs/eq_2res_200hz_4input_ep123/metrics.json"   #3.34875253661426 / 3.789730 / 0.133533958759
# json_file = "batch_test_outputs/eq_2res_200hz_4input_0.5pertb_ep165/metrics.json"   #3.25278 / 3.631963 / 0.11838

# json_file = "batch_test_outputs/idso3_eq_2res_200hz_2input_mselss/metrics.json"   #9.498565 / 10.4504075406 /  0.428831
# json_file = "batch_test_outputs/idso3_eq_2res_200hz_2input_mselss_ep200/metrics.json"   #9.190897 / 10.04809121 /  0.420456

# json_file = "batch_test_outputs/eq_2res_200hz_2input_mselss_ep200/metrics.json"   #4.0465 / 4.6144376475 / 0.20851

##fixed EQUIVARIANT
# json_file = "./batch_filter_outputs_uf20/eq_2res_200hz_3input_samelss_yesbias2_ep200/metrics.json"  # 23.5512
# json_file = "./batch_filter_outputs_uf20/eq_2res_300hz_2input_samelss_yesbias_ep200/metrics.json"  # 23.5512
# json_file = "batch_test_outputs/eq_2res_300hz_2input_samelss_yesbias_ep200/metrics.json"   #3.691 / 4.25427 / 0.2146

# json_file = "batch_test_outputs/eq_2res_100hz_3input_samelss_yesbias_ep200/metrics.json"   #3.32843 / 3.73185 / 0.187510
# json_file = "batch_test_outputs/eq_2res_100hz_3input_samelss_nobias_ep200/metrics.json"   #1.8118 / 1.941 / 0.00449
# json_file = "batch_test_outputs/eq_2res_200hz_3input_samelss_yesbias_ep200/metrics.json"   #3.66583 / 4.1460 / 0.1522
# json_file = "batch_test_outputs/eq_2res_100hz_3input_samelss_yesbias2_ep200/metrics.json"   #2.397 / 2.6358 / 0.103835
# json_file = "batch_test_outputs/eq_2res_100hz_3input_samelss_yesbias5_ep200/metrics.json"   #2.9034 / 3.244 / 0.1418
# json_file = "batch_test_outputs/eq_2res_200hz_3input_samelss_yesbias2_ep200/metrics.json"   #3.2074 / 3.586 / 0.1272
# json_file = "batch_test_outputs/eq_2res_100hz_3input_samelss_nobias2_ep200/metrics.json"   #3.195 / 3.590 / 0.1804

# json_file = "batch_test_outputs/eq_2res_400hz_2input_samelss_yesbias_ep200/metrics.json"   #3.96631392 / 4.529012 / 0.21648

# json_file = "batch_test_outputs/eq_2res_400hz_2input_samelss_nobias_ep200/metrics.json"   #3.9687 / 4.4396736 / 0.254044
# json_file = "batch_test_outputs/eq_2res_300hz_2input_samelss_nobias_ep200/metrics.json"   #3.743494 / 4.265 / 0.2429
# json_file = "batch_test_outputs/eq_2res_200hz_2input_samelss_yesbias_ep200/metrics.json"   #3.7250 / 4.2255 / 0.2117
# json_file = "batch_test_outputs/eq_2res_20hz_2input_samelss_yesbias_ep200/metrics.json"   #6.0420121 / 6.643293 / 0.369866
# json_file = "batch_test_outputs/eq_2res_100hz_2input_samelss_yesbias_ep200/metrics.json"   #3.56921 / 4.07003 / 0.2238
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_ep200/metrics.json"   #2.813683 / 3.0658539 / 0.0627856017
# json_file = "batch_test_outputs/eq_2res_300hz_2input_samelss_yesbias2_ep200/metrics.json"   #3.6960 / 4.252 / 0.2100408
# json_file = "batch_test_outputs/eq_2res_300hz_2input_samelss_yesbias2_ep200_ep199/metrics.json"   #3.58476 / 4.1434 / 0.20719
# json_file = "batch_test_outputs/eq_2res_300hz_2input_samelss_yesbias3_ep200_ep195/metrics.json"   #3.5110 / 4.04015 / 0.2091
# json_file = "batch_test_outputs/eq_2res_300hz_2input_samelss_yesbias3_ep200_ep199/metrics.json"   #3.418 / 3.92807 / 0.20915  << 이거 좋다
json_file = "batch_test_outputs/world_disp_eq_2res_200hz_2input_samelss_yesbias_ep200/metrics.json"   #10.542894 / 3.92807 / 0.20915  << 이거 좋다
json_file = "batch_test_outputs/world_disp_eq_4res_200hz_2input_samelss_yesbias_ep200/metrics.json"   #10.542894 / 3.92807 / 0.20915  << 이거 좋다

json_file = "batch_test_outputs/res_2res_200hz_2input_bd_last_align_pertb_grav_ep200/metrics.json"   #2.3501 / 2.767957 / 0.51820
json_file = "batch_test_outputs/res_2res_200hz_2input_bd_last_align_ep200/metrics.json"   #2.3050622 / 2.740349
json_file = "batch_test_outputs/res_2res_200hz_2input_bd_last_align_pertb_grav_ep195/metrics.json"   #2.601655 / 3.06800
json_file = "batch_test_outputs/res_2res_200hz_2input_bd_last_align_ep195/metrics.json"   # 2.3010 / 2.75464267
json_file = "batch_test_outputs/resnet_2/metrics.json"   # 2.886/ 3.40108 / 0.52671
json_file = "batch_test_outputs/res_4res_200hz_2input_bd_last_align_comp_ep119/metrics.json"   # 2.097839 / 2.494531
json_file = "batch_test_outputs/eq_4res_300hz_2input_bd_last_align_comp_ep200/metrics.json"   # 2.416263 / 2.8061448
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_comp_ep200/metrics.json"   # 2.395914 / 2.7940844
json_file = "batch_test_outputs/eq_4res_100hz_2input_bd_last_align_comp_ep200/metrics.json"   # 2.8114 / 3.191390
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep200_sunny/metrics.json"   # 2.55267600 / 2.994154
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.365/ 2.80415 / 0.18345
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep200_ep199/metrics.json"   # 2.330715135/ 2.73372 / 0.18345
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep300/metrics.json"   # 2.232046/ 2.662174 
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep400/metrics.json"   # 2.332062/ 2.747627 
json_file = "batch_test_outputs/eq_4res_bias_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.5995554/ 3.06401277    
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep200_detach10/metrics.json"   # 2.52480/ 2.96488220    
json_file = "batch_test_outputs/eq_4res_bias_200hz_2input_bd_last_align_ep400/metrics.json"   # 2.464533/ 2.9294992    
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep300_detach10/metrics.json"   # 2.45936/ 2.88416    

json_file = "batch_test_outputs/eq_4res_200hz_3input_bd_ep200_train_bias_2_detach10/metrics.json"   # 0.639315/ 0.68828
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_ep175_train_bias_2_detach10/metrics.json"   # 2.9440270/ 3.437778

json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_ep200_train_bias_2_detach10/metrics.json"   # 3.00238/ 3.508007
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep200_train_bias_2_detach10/metrics.json"   # 3.208/ 3.67237

json_file = "batch_test_outputs/eq_4res_200hz_3input_bd_last_align_ep200_train_bias_4_std_01_detach10/metrics.json"   # 0.4670/ 0.50788
json_file = "batch_filter_outputs_riekf_no_correction_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 6.89836/ 0.33
json_file = "batch_filter_outputs_riekf_uf20/eq_4res_200hz_2input_bd_last_align_ep200_train_bias_2_detach10/metrics.json"   # 3.500170/ 0.1985

json_file = "batch_filter_outputs_riekf_uf20/collect_v_init_bias_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 4.236003/ 0.28632
json_file = "batch_filter_outputs_riekf_uf20/collect_v_from_gt_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 3.00179/ 0.17508
json_file = "batch_filter_outputs_riekf_uf20/collect_v_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 3.00179/ 0.17508
json_file = "batch_filter_outputs_riekf_uf20/collect_v_use_gtR_T_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.997/ 0.1756
# json_file = "batch_filter_outputs_riekf_uf20/collect_v_use_gtR_T_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.99707/ 0.175692 / 0.1756

json_file = "batch_filter_outputs_riekf_const_cov_uf20/no_bias_update_zero_bias_uf20/eq_4res_200hz_2input_bd_last_align_ep200_detach10/metrics.json"   # 3.22479212/ 0.17843
json_file = "batch_filter_outputs_riekf_const_cov_uf20/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 3.0839142/ 0.1672975

json_file = "batch_filter_outputs_riekf_uf1000/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.111517/ 0.14589

json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep300_detach10/metrics.json"   # 2.365/ 2.80415 / 0.18345
json_file = "batch_filter_outputs_riekf_uf20/eq_4res_200hz_2input_bd_last_align_ep300_detach10/metrics.json"   # 3.03616/ 0.28632
json_file = "./batch_test_outputs/eq_4res_200hz_2input_bd_last_align_comp_ep99/metrics.json"  #2.7568 / 3.1515
json_file = "./batch_test_outputs/res_4res_200hz_2input_bd_last_align_comp_ep99/metrics.json"  #2.3553109 / 2.766945
json_file = "./batch_test_outputs/res_4res_200hz_2input_bd_last_align_comp_ep200/metrics.json"  #2.3553109 / 2.766945

json_file = "batch_filter_outputs_riekf_uf200/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.88226/ 0.1603645
json_file = "batch_filter_outputs_riekf_maha_uf200/inno_sigma_2.5_const_N_uf200/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.883687/ 0.1607643
json_file = "batch_filter_outputs_riekf_maha_uf200/inno_sigma_3_const_N_uf200/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.883687/ 0.1607643
json_file = "batch_filter_outputs_riekf_uf1000/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.88067/ 0.18925

json_file = "./batch_test_outputs/eq_4res_200hz_3input_bd_cl_bias_4_cl_deci_10/metrics.json"  #1.605 / 1.7527991
json_file = "./batch_test_outputs/eq_4res_200hz_3input_bd_cl_bias_4_cl_deci_5/metrics.json"  #1.605 / 1.7527991

json_file = "./batch_test_outputs/ln_last_align_2regress_ep199/metrics.json"  # 3.272 /3.679
json_file = "./batch_test_outputs/ln_last_align_2regress_ep199/metrics.json"  # 3.272 /3.679
json_file = "./batch_test_outputs/eq_4res_200hz_3input_bd_last_align_ep400_velbias_8/metrics.json"  # 0.61737 /0.6651
json_file = "./batch_test_outputs/eq_4res_200hz_3input_bd_last_align_ep200_velbias_30/metrics.json"  # 2.08799 /2.2920
json_file = "batch_test_outputs/res_4res_200hz_2input_bd_last_align_comp_ep119/metrics.json"   # 2.0978 / 2.494531 / 1.117250

json_file = "batch_test_outputs/ln_conv_last_align_2regress_nochannelmix/metrics.json"   # 2.508/ 2.9569976 / 1.243664
json_file = "batch_test_outputs/ln_conv_last_align_2regress_nochannelmix_detach_liebracket_lass_ep200/metrics.json"   # 2.666/ 3.063705 / 1.305294
json_file = "batch_test_outputs/ln_conv_last_align_2regress_nochannelmix_mselss_lr_3_AdamW_wdk5_slope_1/metrics.json"   # 2.2856/ 2.7027632 / 1.1854823
json_file = "batch_test_outputs/ln_conv_last_align_2regress_onechannelmix_mselss_lr_2_AdamW_wdk4_slope2_pertbgrav/metrics.json"   # 2.3466815/ 2.751224 / 1.20517
json_file = "batch_test_outputs/ln_conv_last_align_2regress_onechannelmix_mselss_ep80/metrics.json"   # 2.662969
json_file = "batch_test_outputs/ln_conv_last_align_comp_2regress_nochannelmix_slope_1_onechannelmix_mselss_lr_3_AdamW_wdk5_pertbgrav/metrics.json"   # 2.2555502 / 2.62828 / 1.16150 >> 현재까지 제일 좋음
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_comp_ep400_param_2/metrics.json"   # 2.26934173 / 2.66111 / 1.15366476
json_file = "batch_test_outputs/ln_conv_last_align_2regress_onechannelmix_mselss/metrics.json"   # 2.402735/ 2.7890 /1.192899089
json_file = "batch_test_outputs/ln_conv_last_align_2regress_nochannelmix_largechannel_mselss_lr_3_AdamW_wdk5/metrics.json"   # 2.2848/ 2.67624 /1.16008134
json_file = "batch_test_outputs/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_onechannelmix_mselss_lr_3_AdamW_wdk5_pertbgrav/metrics.json"   #  2.585/ 2.9943 /1.3466738
json_file = "batch_test_outputs/eq_4res_200hz_3input_bd_last_align_ep200_velbias_20_pertbgrarv/metrics.json"   #  1.150622/ 1.248984 /0.57633
json_file = "batch_test_outputs/eq_4res_200hz_2input_bd_last_align_ep200/metrics.json"   # 2.36595/ 2.8041553548 / 1.1864
json_file = "batch_test_outputs/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_onechannelmix_mselss_lr_3_AdamW_wdk5_pertbgrav/metrics.json"   #  2.585/ 2.9943 /1.3466738
json_file = "batch_test_outputs/eq_4res_200hz_3input_bd_last_align_ep200_velbias_20_pertbgrav/metrics.json"   #  1.25502691/ 1.379854 /0.603301
json_file = "batch_test_outputs/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_lessbias_deci1/metrics.json"   #  2.27775/ 2.66945 /1.1989515
json_file = "batch_test_outputs/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_lessbias/metrics.json"   #  2.27775/ 2.66945 /1.1989515
json_file = "batch_test_outputs_so3/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_lessbias/metrics.json"   #  2.3303871795/ 2.7389335 /1.1971992
json_file = "batch_test_outputs/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_onechannelmix_mselss_lessbias_2/metrics.json"   #  2.81081/ 3.207540 /1.433300
json_file = "batch_test_outputs/ln_conv_last_align_notcomp_2regress_nochannelmix_slope_1_mselss_lessbias_2/metrics.json"   #  2.273992041

# json_file = "batch_test_outputs/models-resnet/metrics.json"   # 2.8645704/ 1.51938


# json_file = "./batch_filter_outputs_riekf_uf20/no_bias_update_zero_bias_uf20/eq_4res_200hz_2input_bd_last_align_ep200_detach10/metrics.json"  #3.16589884 / 0.1889633

# json_file = "batch_test_outputs/so3_eq_2res_300hz_2input_samelss_yesbias_ep200/metrics.json"   #8.188934 / 9.03163 / 0.37481
# json_file = "batch_test_outputs/so3_eq_2res_300hz_2input_samelss_nobias_ep200/metrics.json"   #3.7434 / 4.26561 / 0.24290
# json_file = "batch_test_outputs/so3_eq_2res_400hz_2input_samelss_nobias_ep200/metrics.json"   #3.96879 / 4.439 / 0.2540
# json_file = "batch_test_outputs/so3_eq_2res_400hz_2input_samelss_nobias_ep200/metrics.json"   #3.96879 / 4.439 / 0.2540
# json_file = "batch_test_outputs/so3_eq_2res_200hz_3input_samelss_nobias_ep200/metrics.json"   #8.188934 / 9.03163 / 0.37481

# json_file = "batch_test_outputs/eq_2res_20hz_3input_mselss_ep200/metrics.json"   #3.487646 / 3.9583 / 0.19888337
# json_file = "batch_test_outputs/eq_2res_20hz_3input_mselss_ep98/metrics.json"   #4.2050 / 4.7047008 / 0.2315928
# json_file = "batch_test_outputs/3input_eq_2res_bd_20hz/metrics.json"   #1.09794 / 1.208837 / 0.07633750

# json_file = "batch_test_outputs/so3_eq_2res_20hz_3input_samelss_nobias_ep200/metrics.json"   #0.4354991 / 0.4662 / 0.00464
# json_file = "batch_test_outputs/so3_eq_2res_100hz_3input_samelss_nobias_ep200/metrics.json"   #1.814 / 1.943818 / 0.004490
# json_file = "batch_test_outputs/so3_eq_2res_200hz_3input_samelss_nobias_ep200/metrics.json"   #2.631 / 2.8413 / 0.00558

# json_file = "batch_test_outputs/eq_2res_20hz_3input_samelss_nobias_ep200/metrics.json"   #0.43981 / 0.471010 / 0.004637   
# json_file = "batch_test_outputs/eq_2res_100hz_3input_samelss_nobias_ep200/metrics.json"   #1.811829 / 1.94102861 / 0.0044
# json_file = "batch_test_outputs/eq_2res_200hz_3input_samelss_nobias_ep200/metrics.json"   #2.63860 / 2.8490 / 0.005586
##fixed EQUIVARIANT

# json_file = "batch_test_outputs/eq_2res_200hz_3input/metrics.json"   #3.4788114 / 3.94063
# json_file = "batch_test_outputs/eq_2res_200hz_3input_ep_more2/metrics.json"   #3.52228753 / 3.938439400
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss/metrics.json"   #3.32894499 / 3.71997
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_less_v_pertb/metrics.json"   #3.111072 / 3.425042299
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_less_v_pertb_ep200/metrics.json"   #3.3402726 / 3.65995785
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb/metrics.json"   #3.1371065 / 3.4769123
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_nodetach/metrics.json"   #5.0732262 / 5.71939  
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_0.5_pertb_ep_more2/metrics.json"   #2.813362 / 3.060839 / 0.068971 <---- 이게 제일 좋은듯
# json_file = "batch_test_outputs/eq_2res_200hz_3input_mselss_no_pertb_atall_ep200/metrics.json"   #2.6725 / 2.886775  
# json_file = "batch_test_outputs/eq_1res_200hz_3input_mselss_no_pertb_atall_ep200/metrics.json"   #2.6276942 /  2.8366642738  

# json_file = "batch_test_outputs/june_sim_resnet/metrics.json"   #7.2944 /  8.007938580  / 1.37
# json_file = "batch_test_outputs/june_sim_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #8.946404721302 /  9.850239  / 0.379029
# json_file = "batch_test_outputs/june_sim_eq_2res_200hz_3input_mselss/metrics.json"   #9.638738 /  10.62049026  / 0.3616468956
# json_file = "batch_test_outputs/sim_world_idso2_2_worldframe/metrics.json"   #22.987536403847 /  25.737584164400
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   # 8.946404721302956 / 9.85023980723 /  0.379029
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_mselss/metrics.json"   # 9.6387 / 10.6204 /  0.3616468
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_samelss_TLIO_previous_400dataset/metrics.json"   # 9.6387 / 10.6204 /  0.3616468

# json_file = "batch_test_outputs/sim_resnet-targ_v/metrics.json"   # 7.294 / 8.007938580 /  1.37263
# json_file = "batch_test_outputs/sim_world_idso3_2_worldframe/metrics.json"   #354.1185 / 408.01704 /  5.799109333753586

# json_file = "batch_test_outputs/sim_body/metrics.json"   #11.3146 /  12.72
# # json_file = "batch_test_outputs/sim_body_ep99/metrics.json"   #10.77844 /  12.0627677
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_mselss/metrics.json"   #12.521605 /  13.8155  / 0.8306772
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_mselss_vino/metrics.json"   #12.91992 /  14.352455  / 0.78445325

# json_file = "batch_test_outputs/sim_body_idso2/metrics.json"   #13.4334099 /  15.146993
# json_file = "batch_test_outputs/sim_idso2_eq_2res_200hz_3input_samelss≠≠_TLIO/metrics.json"   #14.1992214182 /  15.876565 / 0.8945234946906566
# json_file = "batch_test_outputs/sim_idso2_eq_2res_200hz_3input_mselss/metrics.json"   #15.183314 /  16.98737712 / 0.843082719

# json_file = "batch_test_outputs/sim_body_idso3/metrics.json"   #80.197805 /  92.912570675
# json_file = "batch_test_outputs/sim_idso3_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #44.121510 /  50.42309706628    / 1.06438146
# json_file = "batch_test_outputs/sim_idso3_eq_2res_200hz_3input_mselss/metrics.json"   #47.724336 /  54.6730013 / 0.9397254191339016
# json_file = "batch_test_outputs/sim_eq_2res_200hz_3input_samelss_TLIO/metrics.json"   #12.6049 / 14.20915
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
rmse_values = []
rmse_vel_values = []

for key, value in data.items():
    # if key in ("2100421890282669", "2142509999304237", "2142577795957193"):
    #     continue
    
    if "filter" in json_file:
        filter_dict = value.get("filter")
    else:
        filter_dict = value.get("ronin")
    
    if filter_dict:
        ate_value = filter_dict.get("ate")
        if "filter" in json_file:
            rmse_value = filter_dict.get("rpe_rmse_1000")
            rmse_vel_value = filter_dict.get("rpe_rmse_1000")
        else:
            rmse_value = filter_dict.get("rmse")
            rmse_vel_value = filter_dict.get("rpe")
        
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
