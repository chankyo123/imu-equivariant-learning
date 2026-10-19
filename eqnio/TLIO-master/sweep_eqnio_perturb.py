"""
Sweep EqNIO O(2) filter+network under (perturb_type, perturb_deg) configurations.

For each (model, perturb_type, perturb_deg):
  - run main_filter.py once (loops through SUBSET_SIZE sequences internally)
  - write filter outputs to out_dir/<seq>/not_vio_state.txt
  - then call EKF_output_metrics.py to compute filter ATE
Network ATE is also collected via a parallel main_net.py run that reuses the same perturbation flags.

Outputs (SWEEP_OUT, default /tmp/eqnio_sweep, override with --sweep_out):
  <sweep_out>/<model>__<ptype>_<deg>/  -- filter raw state files
  <sweep_out>/results_filter.csv       -- aggregated filter ATE table (aggregate_sweep*.py)
  <sweep_out>/results_network.csv      -- aggregated network ATE table

Ported from Server B. Machine-specific paths are now derived from the repository root:
  TLIO_DATA  = <repo>/local_data/tlio_golden      (override: --tlio_data)
  EQNIO_SRC  = <repo>/eqnio/TLIO-master/src
  PY         = sys.executable                     (override: --python)
  MODELS_TLIO entries point to <repo>/checkpoints/... ; edit the list for other checkpoints.
  Each model dir must contain model_torchscript.pt, checkpoint_best.pt and parameters.json.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TLIO_DATA = os.path.join(REPO_ROOT, "local_data", "tlio_golden")
EQNIO_SRC = os.path.join(REPO_ROOT, "eqnio", "TLIO-master", "src")
PY = sys.executable
SWEEP_OUT = "/tmp/eqnio_sweep"

# (label, ckpt_dir). Server B swept: tlio_o2, tlio_o2_so3aug_80ep, ln_189, resnet_so3aug and
# tlio_resnet_baseline_w1sec_e100. The shipped checkpoints (checkpoints/MANIFEST.md) cover the first,
# third and fourth; add the others here if you retrain them.
CKPT = os.path.join(REPO_ROOT, "checkpoints")
MODELS_TLIO = [
    ("tlio_o2",          f"{CKPT}/tlio_eqnio_o2"),
    # ("tlio_o2_so3aug",   f"{CKPT}/tlio_eqnio_o2_so3aug_80ep"),        # not shipped
    ("ln_189",           f"{CKPT}/tlio_ginio_w_ln189"),
    ("resnet_so3aug",    f"{CKPT}/tlio_resnet_so3aug"),
    # ("tlio_resnet_baseline", f"{CKPT}/tlio_resnet_baseline_w1sec_e100"),  # not shipped
]

PERTURB_DEGS = [3.0, 5.0, 10.0, 15.0, 20.0]
PERTURB_DEGS_SO3 = [30.0, 60.0, 90.0, 120.0, 180.0]


def build_configs(include_so3=False):
    """One (model, perturb_type, deg) tuple per filter run."""
    cfgs = []
    for model_name, model_dir in MODELS_TLIO:
        cfgs.append((model_name, model_dir, "none", 0.0))
        for ptype in ("gravity", "lfa"):
            for deg in PERTURB_DEGS:
                cfgs.append((model_name, model_dir, ptype, deg))
        if include_so3:
            for deg in PERTURB_DEGS_SO3:
                cfgs.append((model_name, model_dir, "so3", deg))
    return cfgs


def slug(model_name, ptype, deg):
    return f"{model_name}__{ptype}_{deg:g}"


def make_subset_list(src_test_list, dst_path, k):
    with open(src_test_list) as f:
        lines = [s.strip() for s in f.readlines() if s.strip()]
    chosen = lines[:k]
    with open(dst_path, "w") as f:
        for s in chosen:
            f.write(s + "\n")
    return chosen


def run_filter(args_tuple):
    """One filter sweep: runs main_filter.py for one config, all sequences inside."""
    model_name, model_dir, ptype, deg, root_dir, test_list_name, out_root, idx, total = args_tuple
    out_dir = os.path.join(out_root, slug(model_name, ptype, deg))
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        PY, f"{EQNIO_SRC}/main_filter.py",
        "--root_dir", root_dir + "/",
        "--model_path", f"{model_dir}/model_torchscript.pt",
        "--model_param_path", f"{model_dir}/parameters.json",
        "--out_dir", out_dir,
        "--test_list", test_list_name,
        "--test_perturb_type", ptype,
        "--test_perturb_deg", str(deg),
        "--erase_old_log",
        "--cpu",
    ]
    t0 = time.time()
    log_path = os.path.join(out_dir, "sweep.log")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    msg = f"[{idx}/{total}] {slug(model_name, ptype, deg):40s} {status} {dt:6.1f}s"
    print(msg, flush=True)
    return out_dir, proc.returncode


def run_net(args_tuple):
    """One network sweep: runs main_net.py with same perturbation."""
    model_name, model_dir, ptype, deg, root_dir, test_list_name, out_root, arch, idx, total = args_tuple
    out_dir = os.path.join(out_root, "net_" + slug(model_name, ptype, deg))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        PY, f"{EQNIO_SRC}/main_net.py",
        "--mode", "test",
        "--root_dir", root_dir,
        "--test_list", test_list_name,
        "--model_path", f"{model_dir}/checkpoint_best.pt",
        "--out_dir", out_dir,
        "--arch", arch,
        "--batch_size", "128", "--workers", "2",
    ]
    if ptype == "lfa":
        cmd += ["--lfa_perturb_max_deg", str(deg)]
    elif ptype == "gravity":
        cmd += [
            "--dataloader_bias_gravity_aug",
            "--perturb_gravity_theta_range", str(deg),
            "--accel_bias_range", "0", "--gyro_bias_range", "0",
        ]
    elif ptype == "so3":
        cmd += ["--so3_test_deg", str(deg)]
    t0 = time.time()
    log_path = os.path.join(out_dir, "sweep_net.log")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    msg = f"[NET {idx}/{total}] {slug(model_name, ptype, deg):40s} {status} {dt:6.1f}s"
    print(msg, flush=True)
    return out_dir, proc.returncode


def main():
    global SWEEP_OUT, TLIO_DATA, PY
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["filter", "network", "both"], default="both")
    parser.add_argument("--seqs", type=int, default=10, help="# sequences to sample from test_list")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset", choices=["tlio"], default="tlio")
    parser.add_argument("--include_so3", action="store_true")
    parser.add_argument("--only_so3", action="store_true",
                        help="Skip non-so3 configs (assumes prior runs covered them)")
    parser.add_argument("--sweep_out", default=SWEEP_OUT, help="output root for all sweep runs")
    parser.add_argument("--tlio_data", default=TLIO_DATA, help="TLIO golden root")
    parser.add_argument("--python", default=PY, help="python interpreter used for the sub-processes")
    args = parser.parse_args()

    SWEEP_OUT, TLIO_DATA, PY = args.sweep_out, args.tlio_data, args.python
    missing = [d for _, d in MODELS_TLIO if not os.path.isdir(d)]
    if missing:
        print("WARNING: model dirs not found (edit MODELS_TLIO):\n  " + "\n  ".join(missing), flush=True)

    os.makedirs(SWEEP_OUT, exist_ok=True)
    root_dir = TLIO_DATA
    src_test = os.path.join(root_dir, "test_list.txt")
    test_list_name = f"sweep_list_n{args.seqs}.txt"
    chosen = make_subset_list(src_test, os.path.join(root_dir, test_list_name), args.seqs)
    print(f"Using {len(chosen)} sequences: {chosen[:3]}... ({test_list_name})", flush=True)

    cfgs = build_configs(include_so3=args.include_so3 or args.only_so3)
    if args.only_so3:
        cfgs = [c for c in cfgs if c[2] == "so3"]
    total = len(cfgs)
    print(f"Total configs: {total} (modes: {args.mode}, include_so3={args.include_so3}, only_so3={args.only_so3})", flush=True)

    if args.mode in ("filter", "both"):
        print("=== FILTER SWEEP ===", flush=True)
        filter_args = [(m, mdir, p, d, root_dir, test_list_name, SWEEP_OUT, i + 1, total)
                       for i, (m, mdir, p, d) in enumerate(cfgs)]
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(run_filter, filter_args))

    if args.mode in ("network", "both"):
        print("=== NETWORK SWEEP ===", flush=True)
        # Read arch from each model's parameters.json
        import json
        net_args = []
        for i, (m, mdir, p, d) in enumerate(cfgs):
            arch = json.load(open(f"{mdir}/parameters.json"))["arch"]
            net_args.append((m, mdir, p, d, root_dir, test_list_name, SWEEP_OUT, arch, i + 1, total))
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(run_net, net_args))

    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
