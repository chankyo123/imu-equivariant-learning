# Paper artefact: NanoBench scenario evaluation (Standard / Aggressive / ID-SO(3) physical remount / calibration
# mismatch) of the Air-IO family; produced scenarios_metrics.json. It is NOT the generator of the Table-2 ID/SO(3)
# column (its remount numbers differ from the paper's; see README "known provenance notes").
"""
REAL Nanobench scenario evaluation.

For each scenario x model:
  1. Create perturbed test data in a work dir (modify gyro/acc/quat/pos as appropriate)
  2. Run Air-IO inference_motion.py on the perturbed data
  3. Read net_output.pickle and compute ATE/RMSE/RPE/drift

Scenarios:
  Standard:        no perturbation (use existing predictions)
  Aggressive:      subset to fast/random sequences
  ID-SO(3):        apply random body-frame rotation per seq (rotate IMU + quat + pos + vel by R)
  Calib-mismatch:  add gyro/acc bias to IMU per seq
"""
import os
import sys
import json
import pickle
import re
import shutil
import subprocess
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (AIRIO_ROOT, CKPT_ROOT, DATA_WORLD, OUTPUT_DIR, TEST_LIST_FILE,  # noqa: E402
                    airio_ckpt_dir, insert_paths, read_test_list)
insert_paths(AIRIO_ROOT)
from utils import CPU_Unpickler  # noqa: E402  (airio/utils)

NANO_ROOT_AWS = str(DATA_WORLD)
AIRIO_ROOT = str(AIRIO_ROOT)
OUT_DIR = str(OUTPUT_DIR)
WORK_DIR = "/tmp/ginio_scenarios"   # perturbed copies of the 17 test sequences + temp configs (original: /tmp/nano_<scenario>)

# exp_dir is where inference_motion.py reads <exp_dir>/ckpt/best_model.ckpt and writes net_output.pickle.
# The first four entries are the models of the original run (the two 0.273M "VN" originals are not
# released); the two "small" entries are the released GINIO-A models of Table 2.
MODELS = [
    {"label": "AirIO no-att",    "exp_dir": "experiments/nanobench_noatt/motion_body_noatt",        "config": "configs/Nanobench/motion_body_noatt.conf"},
    {"label": "AirIO with-att",  "exp_dir": "experiments/nanobench/motion_body_rot",                "config": "configs/Nanobench/motion_body_rot.conf"},
    {"label": "VN no-att",       "exp_dir": "experiments/nanobench_noatt_vn/motion_body_noatt_vn",  "config": "configs/Nanobench/motion_body_noatt_vn.conf"},
    {"label": "VN with-att",     "exp_dir": "experiments/nanobench_rot_vn/motion_body_rot_vn",      "config": "configs/Nanobench/motion_body_rot_vn.conf"},
    {"label": "VN no-att (small)",   "exp_dir": "experiments/nanobench_noatt_vn_small/motion_body_noatt_vn_small", "config": "configs/Nanobench/motion_body_noatt_vn_small.conf"},
    {"label": "VN with-att (small)", "exp_dir": "experiments/nanobench_rot_vn_small/motion_body_rot_vn_small",     "config": "configs/Nanobench/motion_body_rot_vn_small.conf"},
]

TEST_SEQS = read_test_list(TEST_LIST_FILE)
AGGRESSIVE_SEQS = [s for s in TEST_SEQS if "fast" in s or "random_waypoints" in s]


# -- Data perturbation ----------------------------------------------------------
def perturb_seq(seq, scenario, rng, params, nano_root=None):
    """Read original npy, apply scenario perturbation, return new array + transform info."""
    nano_root = NANO_ROOT_AWS if nano_root is None else nano_root
    src = f"{nano_root}/{seq}/imu0_resampled.npy"
    arr = np.load(src).copy()
    # cols: ts_us(1), gyr_W(3), acc_W(3), quat_xyzw(4), pos_W(3), vel_W(3)
    info = {}
    if scenario == "id_so3":
        # ID-SO(3) test: simulate IMU mounted at different orientation R_mount.
        # Effect:
        #   - body-frame IMU rotates by R_mount: body_gyro_new = R_mount @ body_gyro_orig
        #   - world trajectory (pos, vel) UNCHANGED
        #   - quaternion (world<-body): new = orig @ R_mount^T
        # We DON'T modify gyr_W / acc_W / pos_W / vel_W -- only the quaternion.
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(np.pi * 0.25, np.pi * 0.75)  # 45-135 deg
        R_mount = Rotation.from_rotvec(axis * angle).as_matrix()
        # quat: world<-body
        q = arr[:, 7:11]
        rot_orig = Rotation.from_quat(q)
        rot_new = rot_orig * Rotation.from_matrix(R_mount.T)
        arr[:, 7:11] = rot_new.as_quat()
        info["R_mount"] = R_mount.tolist()
    elif scenario == "calib_mismatch":
        # Calibration mismatch: constant bias in BODY frame.
        # body_gyro_new = body_gyro_orig + bias_body
        # In world-frame storage: gyr_W_new[i] = gyr_W[i] + R_i @ bias_body (time-varying)
        gyro_bias_body = rng.normal(scale=params["gyro_bias_std"], size=3)
        acc_bias_body  = rng.normal(scale=params["acc_bias_std"], size=3)
        Rs = Rotation.from_quat(arr[:, 7:11]).as_matrix()  # (N, 3, 3)
        arr[:, 1:4] += np.einsum("nij,j->ni", Rs, gyro_bias_body)
        arr[:, 4:7] += np.einsum("nij,j->ni", Rs, acc_bias_body)
        info["gyro_bias_body"] = gyro_bias_body.tolist()
        info["acc_bias_body"] = acc_bias_body.tolist()
    return arr, info


def prepare_perturbed_data(scenario, seed=42, params=None, nano_root=None, work_dir=WORK_DIR):
    """Create perturbed nanobench data in <work_dir>/nano_<scenario>/."""
    nano_root = NANO_ROOT_AWS if nano_root is None else nano_root
    tmp_root = f"{work_dir}/nano_{scenario}"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)
    rng = np.random.default_rng(seed)
    transform_info = {}
    for seq in TEST_SEQS:
        seq_dir = f"{tmp_root}/{seq}"
        os.makedirs(seq_dir, exist_ok=True)
        # Copy calibration + description files
        for f in ["calibration.json", "imu0_resampled_description.json"]:
            src = f"{nano_root}/{seq}/{f}"
            if os.path.exists(src):
                shutil.copy(src, f"{seq_dir}/{f}")
        # Perturb
        arr, info = perturb_seq(seq, scenario, rng, params or {}, nano_root)
        np.save(f"{seq_dir}/imu0_resampled.npy", arr)
        transform_info[seq] = info
    # Save transformation info for debugging
    with open(f"{tmp_root}/_transforms.json", "w") as f:
        json.dump(transform_info, f, indent=2)
    print(f"  prepared {scenario} data in {tmp_root}")
    return tmp_root


def write_temp_config(model, data_root, scenario_tag, airio_root=None, work_dir=WORK_DIR, ckpt_root=CKPT_ROOT):
    """
    Write a temp config that:
      - points at the perturbed data_root
      - points at a per-scenario output exp_dir so we don't overwrite the original net_output.pickle
      - keeps the config basename matching the original so Air-IO resolves checkpoint paths correctly
    Returns (conf_path, eval_pickle_path) where eval_pickle_path is where Air-IO will write the new predictions,
    or None if no checkpoint exists for the model (neither airio/<exp_dir>/ckpt nor a released folder).
    """
    airio_root = AIRIO_ROOT if airio_root is None else airio_root
    base_conf = f"{airio_root}/{model['config']}"
    base_data = f"{airio_root}/configs/datasets/Nanobench/nanobench_body.conf"
    tmp_conf_dir = f"{work_dir}/_air_io_scenarios/{scenario_tag}"
    os.makedirs(tmp_conf_dir, exist_ok=True)

    # Modified dataset conf (override data_root in every block; the original replaced the
    # hard-coded machine path string with the perturbed root)
    with open(base_data) as f:
        data_conf_text = f.read()
    data_conf_text = re.sub(r"data_root:\s*\S+", f"data_root: {data_root}", data_conf_text)
    tmp_data_conf = f"{tmp_conf_dir}/nanobench_body.conf"
    with open(tmp_data_conf, "w") as f:
        f.write(data_conf_text)

    # Modified main conf: also override general.exp_dir -> per-scenario, so checkpoints come from original
    # but predictions go to a new directory.
    with open(base_conf) as f:
        main_conf_text = f.read()
    main_conf_text = main_conf_text.replace(
        '"../datasets/Nanobench/nanobench_body.conf"',
        f'"{tmp_data_conf}"'
    )

    # Air-IO does: exp_dir = exp_dir_in_conf + conf_name; checkpoints loaded from exp_dir/ckpt/
    # We need checkpoints to load from original location BUT outputs to go to scenario-specific path.
    # Trick: set general.exp_dir to point at the original location (so ckpts found), then write outputs
    # to a *symlinked* alternate. Actually easier: copy the original ckpt to scenario dir.
    orig_exp_dir = f"{airio_root}/{model['exp_dir']}"
    ckpt_dir = airio_ckpt_dir(model['exp_dir'], ckpt_root, airio_root)   # airio/<exp_dir>/ckpt or released folder
    if ckpt_dir is None:
        print(f"  [{model['label']}] no checkpoint at {orig_exp_dir}/ckpt and no released folder for it -- skipping")
        return None
    scenario_exp_dir = f"{work_dir}/_air_io_scenarios/{scenario_tag}_outputs/{os.path.dirname(model['exp_dir'])}"
    os.makedirs(scenario_exp_dir, exist_ok=True)
    # The full exp_dir name expected by Air-IO = {scenario_exp_dir}/{conf_name}/
    # Create symlink to original (or released) ckpt dir
    conf_name = os.path.basename(model["config"]).replace(".conf", "")
    target_sub = f"{scenario_exp_dir}/{conf_name}"
    if not os.path.exists(target_sub):
        os.makedirs(target_sub, exist_ok=True)
    target_ckpt = f"{target_sub}/ckpt"
    if not os.path.exists(target_ckpt):
        os.symlink(str(Path(ckpt_dir).resolve()), target_ckpt)
    # Also need to ensure parameters.yaml exists since Air-IO might read it
    for pyaml in (f"{orig_exp_dir}/parameters.yaml", f"{ckpt_dir}/parameters.yaml"):
        if os.path.exists(pyaml) and not os.path.exists(f"{target_sub}/parameters.yaml"):
            shutil.copy(pyaml, f"{target_sub}/parameters.yaml")

    # Rewrite exp_dir in main conf to scenario path (relative)
    # Original line: "exp_dir: experiments/nanobench_noatt"
    new_exp_dir = scenario_exp_dir
    main_conf_text = re.sub(
        r"exp_dir:\s*\S+",
        f"exp_dir: {new_exp_dir}",
        main_conf_text
    )

    orig_name = os.path.basename(model["config"])
    tmp_main_conf = f"{tmp_conf_dir}/{orig_name}"
    with open(tmp_main_conf, "w") as f:
        f.write(main_conf_text)

    eval_pickle = f"{target_sub}/net_output.pickle"
    return tmp_main_conf, eval_pickle


def run_inference(model, conf_path, eval_pickle, device="cuda:0", airio_root=None):
    """Run Air-IO inference. Returns the eval_pickle path if successful."""
    airio_root = AIRIO_ROOT if airio_root is None else airio_root
    cmd = [
        sys.executable, "inference_motion.py",
        "--config", conf_path,
        "--device", device,
    ]
    print(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=airio_root, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-1500:]}")
        return None
    if not os.path.exists(eval_pickle):
        print(f"  no pickle at {eval_pickle}")
        return None
    return eval_pickle


# -- Metric helpers -------------------------------------------------------------
def umeyama_align(src, dst):
    sm = src.mean(0); dm = dst.mean(0)
    s = src - sm; d = dst - dm
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1; R = Vt.T @ U.T
    return (R @ src.T).T + (dm - R @ sm)

def rpe_at_dt(pp, gg, ts, dt_s=2.0):
    N = len(ts); errs = []
    for i in range(N):
        j = np.searchsorted(ts, ts[i] + dt_s)
        if j >= N or j == i: continue
        errs.append(np.linalg.norm((pp[j]-pp[i]) - (gg[j]-gg[i])))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

def compute_metrics(pp_pos, gg_pos, ts):
    pa = umeyama_align(pp_pos, gg_pos)
    ate = float(np.sqrt(np.mean(np.sum((pa - gg_pos)**2, axis=1))))
    rmse = float(np.sqrt(np.mean(np.sum((pp_pos - gg_pos)**2, axis=1))))
    L = float(np.sum(np.linalg.norm(np.diff(gg_pos, axis=0), axis=1)))
    drift = float(np.linalg.norm(pp_pos[-1] - gg_pos[-1])) / max(L, 1e-6)
    return dict(ATE=ate, RMSE=rmse, RPE2s=rpe_at_dt(pp_pos, gg_pos, ts, 2.0), drift=drift)


def integrate_body_vel(net_vel_body, ts_pred, quat_at_ts, pos0):
    R = Rotation.from_quat(quat_at_ts).as_matrix()
    vW = np.einsum("nij,nj->ni", R, net_vel_body)
    dt = np.diff(ts_pred)
    vmid = 0.5 * (vW[:-1] + vW[1:])
    pos = np.zeros_like(vW)
    pos[1:] = np.cumsum(vmid * dt[:, None], axis=0)
    pos += (pos0 - pos[0])
    return pos


def compute_seq_metrics(seq, net_pkl, data_root):
    """Load predictions + GT, integrate, compute metrics."""
    with open(net_pkl, "rb") as f:
        data = CPU_Unpickler(f).load()
    if seq not in data:
        return None
    d = data[seq]
    ts_pred = (d["ts"].cpu().numpy() if hasattr(d["ts"], "cpu") else np.asarray(d["ts"])).squeeze()
    net_vel = d["net_vel"].cpu().numpy() if hasattr(d["net_vel"], "cpu") else np.asarray(d["net_vel"])
    if net_vel.ndim == 3: net_vel = net_vel[0]

    arr = np.load(f"{data_root}/{seq}/imu0_resampled.npy")
    ts_imu = arr[:, 0] / 1e6
    quat = arr[:, 7:11]
    pos = arr[:, 11:14]
    idxs = np.searchsorted(ts_imu, ts_pred).clip(0, len(ts_imu)-1)
    pred_pos = integrate_body_vel(net_vel, ts_pred, quat[idxs], pos[idxs][0])
    return compute_metrics(pred_pos, pos[idxs], ts_pred)


# -- Main -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+",
                    default=["standard", "aggressive", "id_so3", "calib_mismatch"],
                    choices=["standard", "aggressive", "id_so3", "calib_mismatch"])
    ap.add_argument("--models", nargs="+", default=None, choices=[m["label"] for m in MODELS],
                    help="subset of model labels (default: all; models without a checkpoint are skipped)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gyro-bias-std", type=float, default=0.01)   # rad/s
    ap.add_argument("--acc-bias-std", type=float, default=0.10)    # m/s^2
    ap.add_argument("--data_root", default=NANO_ROOT_AWS, help="world-frame NanoBench root (nanobench_processed_2nd_version)")
    ap.add_argument("--airio_root", default=AIRIO_ROOT, help="airio/ (inference_motion.py runs from here)")
    ap.add_argument("--ckpt_root", default=str(CKPT_ROOT), help="released checkpoint folders (fallback when airio/<exp_dir>/ckpt is absent)")
    ap.add_argument("--work_dir", default=WORK_DIR, help="scratch dir for perturbed data, temp configs and scenario pickles")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    models = [m for m in MODELS if args.models is None or m["label"] in args.models]

    results = {m["label"]: {} for m in models}

    for sc in args.scenarios:
        print(f"\n=== Scenario: {sc} ===")
        if sc == "standard":
            data_root = args.data_root
            seqs = TEST_SEQS
        elif sc == "aggressive":
            data_root = args.data_root
            seqs = AGGRESSIVE_SEQS
        else:
            params = {"gyro_bias_std": args.gyro_bias_std, "acc_bias_std": args.acc_bias_std}
            data_root = prepare_perturbed_data(sc, args.seed, params, args.data_root, args.work_dir)
            seqs = TEST_SEQS

        for model in models:
            print(f"\n[{model['label']}] scenario={sc}")
            if sc in ("standard", "aggressive"):
                # Use existing pickle (from inference_motion.py on the unperturbed data; see round1_eval*.py)
                pkl_path = f"{args.airio_root}/{model['exp_dir']}/net_output.pickle"
                eval_root = args.data_root
                if not os.path.exists(pkl_path):
                    print(f"  no pickle at {pkl_path} -> skip")
                    continue
            else:
                # Run perturbed inference
                tc = write_temp_config(model, data_root, scenario_tag=sc, airio_root=args.airio_root,
                                       work_dir=args.work_dir, ckpt_root=args.ckpt_root)
                if tc is None:
                    continue
                conf, eval_pkl = tc
                pkl_path = run_inference(model, conf, eval_pkl, args.device, args.airio_root)
                eval_root = data_root
                if not pkl_path:
                    print(f"  inference failed -> skip")
                    continue
            per_seq = {}
            for seq in seqs:
                try:
                    m = compute_seq_metrics(seq, pkl_path, eval_root)
                    if m: per_seq[seq] = m
                except Exception as e:
                    print(f"    {seq}: ERROR {e}")
            results[model["label"]][sc] = per_seq

            # Aggregate
            if per_seq:
                ates = [v["ATE"] for v in per_seq.values()]
                rmses = [v["RMSE"] for v in per_seq.values()]
                rpes = [v["RPE2s"] for v in per_seq.values() if not np.isnan(v["RPE2s"])]
                drifts = [v["drift"] for v in per_seq.values()]
                print(f"  n={len(per_seq):>2}  ATE mean={np.mean(ates):.3f}  med={np.median(ates):.3f}  "
                      f"RMSE={np.mean(rmses):.3f}  RPE@2s={np.mean(rpes):.3f}  drift={np.mean(drifts):.3f}")

    # Save
    out_json = f"{args.out_dir}/scenarios_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    # Summary table
    lines = ["Nanobench scenarios summary (real evaluation)\n"]
    lines.append(f"{'Model':<22} {'Scenario':<18} {'#':>3}  {'ATE mean':>9} {'ATE med':>8}  {'RMSE':>7}  {'RPE@2s':>7}  {'drift':>7}")
    lines.append("-" * 95)
    for label, scs in results.items():
        for sc in args.scenarios:
            ps = scs.get(sc, {})
            if not ps: continue
            ates = [v["ATE"] for v in ps.values()]
            rmses = [v["RMSE"] for v in ps.values()]
            rpes = [v["RPE2s"] for v in ps.values() if not np.isnan(v["RPE2s"])]
            drifts = [v["drift"] for v in ps.values()]
            lines.append(f"{label:<22} {sc:<18} {len(ps):>3}  {np.mean(ates):>9.3f} {np.median(ates):>8.3f}  "
                         f"{np.mean(rmses):>7.3f}  {np.mean(rpes):>7.3f}  {np.mean(drifts):>7.3f}")
        lines.append("")
    out_txt = f"{args.out_dir}/scenarios_summary.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {out_txt}")


if __name__ == "__main__":
    main()
