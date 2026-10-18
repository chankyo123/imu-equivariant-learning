# Paper artefact: none -- planning stub for the filter-connected covariance ablation (emits a command manifest and
# aggregates metrics.json files); the paper's filter results (Table 1) were not produced with it.
"""Filter-connected covariance-head ablation.

Wraps:
  - TLIO `main_filter.py` (in eqnio/TLIO-master/src) for TLIO-Golden
  - a NanoBench filter driver (`filter_eval.py`, not included in this release) for Nanobench

Compares cov-head variants on CLEAN test inputs and on SO(3)-ROTATED test inputs.
For each (model, condition), we record ATE and the Delta_R NLL = |NLL(R.z, R.y) - NLL(z, y)|.

This script does not execute the filter (heavy job); it produces a manifest of commands
to launch and a downstream --aggregate option that reads the resulting metrics JSON files
to build the main paper CSV.

Usage:
  # 1) generate launch script
  python filter_connected_eval.py --emit-cmds --dataset tlio
  # 2) run those commands on the cluster (~3h CPU per model per condition)
  # 3) aggregate
  python filter_connected_eval.py --aggregate --dataset tlio

Release note: the emitted TLIO commands are the original template. They use the flags
`--model_arch`, `--list_path` and `--so3_rotate_inputs`, which the released
eqnio/TLIO-master/src/main_filter.py does not implement (it needs --model_param_path and selects
the Sigma branch from the torchscript's parameters.json); treat the manifest as a starting point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CKPT_ROOT, EQNIO_SRC, PAPER_DIR, TLIO_GOLDEN, resolve_ckpt_dir  # noqa: E402
from _utils import OUTPUT_DIR, jsonsafe, save_summary  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry (extend as you train more variants)
# ---------------------------------------------------------------------------

def tlio_models(ckpt_root: Path = CKPT_ROOT) -> dict:
    """name : (torchscript path, arch_type string used by meas_source_torchscript).
    Released traces only; the original registry also listed overnight/_existing_vn_so3_t0012
    (`vn_eqnio_so3_spectral_t0012`), which is not released."""
    ckpt_root = Path(ckpt_root)
    return {
        "vn_eqnio_so3_spectral_so3aug": (
            str(resolve_ckpt_dir("ginio_e_so3aug_tlio", ckpt_root) / "model_torchscript.pt"),
            "eq_so3_frame_fullCov_2vec_2deep",  # contains "so3_frame"
        ),
        "eqnio_o2_fullCov_paper": (
            str(resolve_ckpt_dir("eqnio_o2_tlio_ours", ckpt_root) / "model_torchscript.pt"),
            "eq_o2_frame_fullCov",  # contains "tlio_frame_fullCov"
        ),
    }


TLIO_MODELS = tlio_models()

NANOBENCH_MODELS = {
    # Filled in by the user once Nanobench-side torchscripts are converted
}

TLIO_TEST_SEQS_FILE = str(TLIO_GOLDEN / "test_list.txt")
MAIN_FILTER = EQNIO_SRC / "main_filter.py"
NANO_FILTER_EVAL = PAPER_DIR / "filter_eval.py"   # not part of this release


# ---------------------------------------------------------------------------
# Command emission
# ---------------------------------------------------------------------------

def emit_tlio_cmds(out_dir: Path, models: dict, test_seqs_file: str) -> list[str]:
    cmds = []
    for name, (ts_path, arch) in models.items():
        for cond in ("clean", "rotR"):
            tag = f"tlio_{name}_{cond}"
            cmd = dedent(f"""\
            python {MAIN_FILTER} \\
              --model_path {ts_path} \\
              --model_arch {arch} \\
              --list_path {test_seqs_file} \\
              --out_dir {out_dir/tag} \\
              {'--so3_rotate_inputs' if cond == 'rotR' else ''} \\
              --erase_old_log 2>&1 | tee {out_dir/(tag + '.log')}
            """).strip()
            cmds.append(cmd)
    return cmds


def emit_nanobench_cmds(out_dir: Path) -> list[str]:
    if not NANOBENCH_MODELS:
        print("[INFO] NANOBENCH_MODELS empty -- populate first")
        return []
    cmds = []
    for name, (ts_path, arch) in NANOBENCH_MODELS.items():
        for cond in ("clean", "rotR"):
            tag = f"nano_{name}_{cond}"
            cmd = dedent(f"""\
            python {NANO_FILTER_EVAL} \\
              --model {ts_path} \\
              --arch {arch} \\
              --out {out_dir/tag} \\
              {'--so3-rotate' if cond == 'rotR' else ''} 2>&1 | tee {out_dir/(tag + '.log')}
            """).strip()
            cmds.append(cmd)
    return cmds


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def safe_load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def aggregate(dataset: str, out_dir: Path, models: dict) -> dict:
    rows = []
    for name in models:
        per_cond = {}
        for cond in ("clean", "rotR"):
            tag = f"{'tlio' if dataset == 'tlio' else 'nano'}_{name}_{cond}"
            metrics_json = out_dir / tag / "metrics.json"
            md = safe_load_json(metrics_json) or {}
            per_cond[cond] = {
                "ate_mean": md.get("ate_mean"),
                "ate_median": md.get("ate_median"),
                "nll_mean": md.get("nll_mean"),
                "exists": metrics_json.exists(),
            }
        delta_nll = None
        if (per_cond["clean"]["nll_mean"] is not None
                and per_cond["rotR"]["nll_mean"] is not None):
            delta_nll = abs(per_cond["rotR"]["nll_mean"] - per_cond["clean"]["nll_mean"])
        rows.append({
            "model": name,
            "clean": per_cond["clean"],
            "rotR": per_cond["rotR"],
            "delta_R_nll": delta_nll,
        })
    return {"dataset": dataset, "rows": rows}


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("model,clean_ate,rot_ate,clean_nll,rot_nll,delta_R_nll\n")
        for r in rows:
            f.write(
                f"{r['model']},"
                f"{r['clean'].get('ate_mean')},{r['rotR'].get('ate_mean')},"
                f"{r['clean'].get('nll_mean')},{r['rotR'].get('nll_mean')},"
                f"{r['delta_R_nll']}\n"
            )
    print(f"[saved] {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["tlio", "nanobench"], required=True)
    p.add_argument("--emit-cmds", action="store_true",
                   help="emit shell commands to a launch script")
    p.add_argument("--aggregate", action="store_true",
                   help="read per-run metrics.json and write the main CSV")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR / "filter"))
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT), help="released checkpoint folders (model_torchscript.pt)")
    p.add_argument("--tlio-test-list", default=TLIO_TEST_SEQS_FILE)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = tlio_models(args.ckpt_root) if args.dataset == "tlio" else NANOBENCH_MODELS

    if args.emit_cmds:
        cmds = emit_tlio_cmds(out_dir, models, args.tlio_test_list) if args.dataset == "tlio" \
            else emit_nanobench_cmds(out_dir)
        if not cmds:
            print("[WARN] no commands generated")
            return 0
        sh_path = out_dir / f"launch_{args.dataset}.sh"
        with open(sh_path, "w") as f:
            f.write("#!/usr/bin/env bash\nset -e\n\n")
            for c in cmds:
                f.write(c + "\n\n")
        os.chmod(sh_path, 0o755)
        print(f"[saved] {sh_path}  ({len(cmds)} commands)")
        return 0

    if args.aggregate:
        agg = aggregate(args.dataset, out_dir, models)
        csv_path = out_dir / f"main_table_{args.dataset}.csv"
        write_csv(agg["rows"], csv_path)
        text_lines = [f"Filter-connected ablation  dataset={args.dataset}"]
        for r in agg["rows"]:
            text_lines.append(
                f"  {r['model']}: clean_ATE={r['clean'].get('ate_mean')}  "
                f"rot_ATE={r['rotR'].get('ate_mean')}  deltaR_NLL={r['delta_R_nll']}"
            )
        save_summary(args.out_dir, f"main_table_{args.dataset}",
                     jsonsafe(agg), "\n".join(text_lines) + "\n")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
