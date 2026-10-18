import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def unwrap_rpy(rpys):
    diff = rpys[1:, :] - rpys[0:-1, :]
    uw_rpys = np.zeros(rpys.shape)
    uw_rpys[0, :] = rpys[0, :]
    diff[diff > 300] = diff[diff > 300] - 360
    diff[diff < -300] = diff[diff < -300] + 360
    uw_rpys[1:, :] = uw_rpys[0, :] + np.cumsum(diff, axis=0)
    return uw_rpys

def wrap_rpy(uw_rpys, radians=False):
    bound = np.pi if radians else 180
    rpys = uw_rpys
    while rpys.min() < -bound:
        rpys[rpys < -bound] = rpys[rpys < -bound] + 2*bound
    while rpys.max() >= bound:
        rpys[rpys >= bound] = rpys[rpys >= bound] - 2*bound
    return rpys


def compute_rpe(rpe_ns, ps, ps_gt, yaw, yaw_gt):
    ns = ps_gt.shape[0]
    assert ns - rpe_ns > 100
    assert ps.shape == ps_gt.shape
    assert yaw.shape == yaw_gt.shape

    # rpes = []
    relative_yaw_errors = []
    for i in range(0, ns - rpe_ns, 100):
        # chunk = ps[i : i + rpe_ns, :]
        # chunk_gt = ps_gt[i : i + rpe_ns, :]
        chunk_yaw = yaw[i : i + rpe_ns, :]
        chunk_yaw_gt = yaw_gt[i : i + rpe_ns, :]
        initial_error_yaw = wrap_rpy(chunk_yaw[0, :] - chunk_yaw_gt[0, :])
        # final_error_p_relative = Rotation.from_euler(
        #     "z", initial_error_yaw, degrees=True
        # ).as_matrix().dot((chunk[[-1], :] - chunk[[0], :]).T)[0, :, :].T - (
        #     chunk_gt[[-1], :] - chunk_gt[[0], :]
        # )
        final_error_yaw = wrap_rpy(chunk_yaw[[-1], :] - chunk_yaw_gt[[-1], :])
        # rpes.append(final_error_p_relative)
        relative_yaw_errors.append(wrap_rpy(final_error_yaw - initial_error_yaw))
    # rpes = np.concatenate(rpes, axis=0)
    relative_yaw_errors = np.concatenate(relative_yaw_errors, axis=0)

    # plt.figure("relative yaw error")
    # plt.plot(relative_yaw_errors)
    # plt.figure("rpes list")
    # plt.plot(rpes)
    ## compute statistics over z separatly
    # rpe_rmse = np.sqrt(np.mean(np.sum(rpes ** 2, axis=1)))
    # rpe_rmse_z = np.sqrt(np.mean(rpes[:, 2] ** 2))
    relative_yaw_rmse = np.sqrt(np.mean(relative_yaw_errors ** 2))
    return relative_yaw_rmse#rpe_rmse, rpe_rmse_z, relative_yaw_rmse


def calculate_ekf_metrics(exp_list,gt_path_main,output_file_name):
    # exp_list = ['TLIO-master/output/resnet_tlio_local_gravity_aligned_50epochs/ekf_test_rot_aug'
    #             ]
    ## for the table
    result = []
    for exp in exp_list:
        print('results for '+exp)
        delta = int(200*5) ## rte metric same as ronin
        # path = 'TLIO-master/output/'
        dir_path = exp
        ate_list = []
        rte_list = []
        drift_list = []
        aye_list = []
        yaw_drift_list = []
        min_ate = 0
        min_ate_folder =''
        start_idx = 2000 #2s
        if 'rnin-vio' in dir_path:
            start_idx = 0
        count_file=0
        for folder in  os.listdir(dir_path):
            if count_file%25==0:
                print(count_file, folder)
            count_file += 1
            if os.path.isfile(dir_path+'/'+folder):
                continue
            else:
                if os.path.exists(dir_path+'/'+folder+'/not_vio_state.txt') or os.path.exists(dir_path+folder+'/not_vio_state.txt.npy'):
                    # kf_data = np.loadtxt(dir_path+'/'+folder+'/not_vio_state.txt', delimiter=",")
                    txt_path = dir_path+'/'+folder+'/not_vio_state.txt'
                    py_path = dir_path+'/'+folder+'/not_vio_state.txt.npy'
                    if os.path.exists(txt_path):
                        kf_data = np.loadtxt(txt_path, delimiter=",")
                    elif os.path.exists(py_path):
                        kf_data = np.load(py_path)
                    else:
                        continue
                    # gt_path = 'aria_data/downsampled_w_i/'+folder+'/imu0_resampled.npy'
                    # gt_path = 'TLIO-master/local_data/tlio_golden/'+folder+'/imu0_resampled.npy'
                    gt_path = gt_path_main+folder+'/imu0_resampled.npy'
                    # print(folder)
                    gt_data = np.load(gt_path)

                    kf_ts = kf_data[start_idx:, 27]
                    kf_p = kf_data[start_idx:, 12:15]

                    gt_ts = gt_data[:,0]*1e-6
                    gt_p = gt_data[:,-6:-3]

                    ## reading yaw data
                    R_init = kf_data[start_idx, :9].reshape(-1, 3, 3)
                    r_init = Rotation.from_matrix(R_init)
                    Rs = kf_data[start_idx:, :9].reshape(-1, 3, 3)
                    rs = Rotation.from_matrix(Rs)
                    euls = rs.as_euler("xyz", degrees=True)

                    vio_rq = gt_data[:,-10:-6]
                    vio_r = Rotation.from_quat(vio_rq)
                    vio_euls = vio_r.as_euler("xyz", degrees=True)
                    vio_uw_euls = unwrap_rpy(vio_euls)

                    if gt_ts[0] > kf_ts[0]:
                        gt_ts = np.insert(gt_ts, 0, kf_ts[0])
                        gt_p = np.concatenate([gt_p[0].reshape(1, 3), gt_p], axis=0)
                        vio_uw_euls = np.concatenate(
                            [vio_uw_euls[0].reshape(1, 3), vio_uw_euls], axis=0
                        )
                    if gt_ts[-1] < kf_ts[-1]:
                        gt_ts = np.insert(gt_ts, -1, kf_ts[-1])
                        gt_p = np.concatenate([gt_p, gt_p[-1].reshape(1, 3)], axis=0)
                        vio_uw_euls = np.concatenate(
                            [vio_uw_euls, vio_uw_euls[-1].reshape(1, 3)], axis=0
                        )

                    inter_gt_p = interp1d(gt_ts, gt_p, axis=0)(kf_ts)
                    ate = np.sqrt(np.mean(np.linalg.norm(kf_p - inter_gt_p, axis=1) ** 2))
                    
                    ate_list.append(ate)
                    if ate>min_ate:
                        min_ate = ate
                        min_ate_folder = folder
                    # print(np.loadtxt(dir_path+'/'+folder+'/net_outputs.txt', delimiter=",").shape)

                    # rte calculation
                    re_est = inter_gt_p[1:-1-delta:delta] - inter_gt_p[delta:-1:delta]
                    re_gt = kf_p[1:-1-delta:delta] - kf_p[delta:-1:delta]

                    rte_list.append(np.sqrt(np.mean(np.linalg.norm(re_gt - re_est, axis=1)** 2)))


                    ## drift % calculation
                    norm_gt = np.linalg.norm( kf_p[1:] - kf_p[:-1], axis=1)
                    drift_list.append((np.linalg.norm(kf_p[-1] - inter_gt_p[-1])/np.sum(norm_gt))*100)


                        
                    vio_uw_euls_interp = interp1d(gt_ts, vio_uw_euls, axis=0)(kf_ts)
                    vio_euls = wrap_rpy(vio_uw_euls_interp)

                    ## AYE
                    filter_heading_error = wrap_rpy(euls - vio_euls)[:, 2]
                    aye_list.append(np.sqrt(np.nansum(filter_heading_error ** 2)/ np.count_nonzero(~(np.isnan(filter_heading_error)))))

                    ## Yaw drift
                    ts = kf_ts - kf_ts[0]
                    euls[:, 2] = euls[:, 2] - (euls[0, 2] - vio_euls[0, 2])
                    euls = wrap_rpy(euls)
                    angular_drift_filter = np.linalg.norm(euls[-1, 2] - vio_euls[-1, 2])
                    yaw_drift_list.append(angular_drift_filter / ts.max() * 3600 * 1e-3) # because its in ms

        result.append({'exp' :dir_path,'ate':np.array(ate_list).mean(),'rte':np.array(rte_list).mean(),
                    'drift':np.array(drift_list).mean(),'aye':np.array(aye_list).mean(),'yaw_drift':np.array(yaw_drift_list).mean()})

    import pandas as pd
    result_df = pd.DataFrame(result) 
    result_df.to_csv(output_file_name+'.csv')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        type=str,
        action='append',  # Allows multiple uses of the argument
        # required=True,
        help="A file to process (can be used multiple times)",
        # default=['/home/royinakj/TLIO-master/output/resnet_tlio_local_gravity_aligned_50epochs/ekf_output']
    )
    parser.add_argument("--ground_truth_path", type=str, default="/home/royinakj/TLIO-master/local_data/tlio_golden/")
    parser.add_argument("--output_file_name", type=str, default="tlio_aug")

    args = parser.parse_args()

    # exp_list,gt_path,output_file_name
    calculate_ekf_metrics(args.files, args.ground_truth_path, args.output_file_name)
    