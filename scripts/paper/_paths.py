# Shared path resolution for the paper-evaluation scripts (no paper table of its own).
"""Repository-relative paths for ``scripts/paper``.

Every script in this folder derives its inputs from a single repository root so that the
hard-coded research-box paths of the original scripts (``/home/ubuntu/imu-work/...``,
``/home/chankyokim/...``, ``Archive/checkpoints/...``) are gone.  Override points:

  GINIO_ROOT        repository root (default: two levels above this file)
  GINIO_CKPT_ROOT   folder holding the released checkpoint folders (default: <root>/checkpoints)
  GINIO_DATA_ROOT   folder holding the datasets (default: <root>/local_data)
  GINIO_PAPER_OUT   where the scripts write (default: <root>/scripts/paper/outputs)

Checkpoint folders are referred to by their ``checkpoints/MANIFEST.md`` names; the flat
``<benchmark>_<row>_<variant>`` names of the release bundle (``RENAMES.txt``) are accepted as
aliases, see ``CKPT_ALIASES`` / ``resolve_ckpt_dir``.  The mapping from the authors' archive
folder names to the released names lives here as well, so no script needs the old names.
"""
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("GINIO_ROOT", Path(__file__).resolve().parents[2]))
PAPER_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO / "src"                                   # Lie-Neuron pipeline (network.model_factory)
SCRIPTS_DIR = REPO / "scripts"                           # scripts/lfa_preprocessing.py
AIRIO_ROOT = REPO / "airio"                              # Air-IO + VN models (model.net_dict, utils.CPU_Unpickler)
EQNIO_SRC = REPO / "eqnio" / "TLIO-master" / "src"       # EqNIO fork (network, tracker.meas_source_torchscript)

CKPT_ROOT = Path(os.environ.get("GINIO_CKPT_ROOT", REPO / "checkpoints"))
LOCAL_DATA = Path(os.environ.get("GINIO_DATA_ROOT", REPO / "local_data"))
DATA_WORLD = LOCAL_DATA / "nanobench_processed_2nd_version"          # world-frame IMU, gravity removed
DATA_LFA = LOCAL_DATA / "nanobench_bodyframe_2nd_version"            # body-frame copy; name turns LFA on
DATA_NOLFA = LOCAL_DATA / "nanobench_bodyframe_nolfa_2nd_version"    # symlink to the same bytes; name turns LFA off
TLIO_GOLDEN = LOCAL_DATA / "tlio_golden"

OUTPUT_DIR = Path(os.environ.get("GINIO_PAPER_OUT", PAPER_DIR / "outputs"))
TEST_LIST_FILE = PAPER_DIR / "nano_test_list.txt"       # the 17 NanoBench test sequences (= Archive/data/nano_test_list.txt)
REFERENCE_INPUTS = PAPER_DIR / "reference_inputs"

# Release folder name (checkpoints/MANIFEST.md, flat <benchmark>_<row>_<variant> scheme) -> older staging names.
# resolve_ckpt_dir() returns the first of (name, *aliases) that exists under the checkpoint root.
CKPT_ALIASES = {
    "nanobench_airio_noatt": ["airio_noatt_nanobench"],
    "nanobench_airio_withatt": ["airio_withatt_nanobench"],
    "nanobench_ginio_a_noatt": ["ginio_a_noatt_nanobench"],
    "nanobench_ginio_a_withatt": ["ginio_a_withatt_nanobench"],
    "nanobench_ginio_e_so3aug_tlio": ["ginio_e_so3aug_tlio"],
    "tlio_eqnio_o2_ours": ["eqnio_o2_tlio_ours"],
    "nanobench_ginio_rd_t55": ["ginio_rd_nolfa_t55_nanobench"],
    "nanobench_ginio_rs_chol_mse": ["ginio_rs_chol_mse_nanobench"],
    "nanobench_ginio_rs_chol_nll": ["ginio_rs_chol_nll_nanobench"],
    "nanobench_ginio_rd_lfa_mse": ["ginio_rd_lfa_mse_nanobench"],
    "nanobench_ginio_rd_lfa_nll": ["ginio_rd_lfa_nll_nanobench"],
}

# Archive/checkpoints/<old name>  ->  checkpoints/<released folder>   (Supplement Table 9 models)
RELEASED_LN = {
    "ln_resnet_lfa_MSE": "nanobench_ginio_rd_lfa_mse",
    "ln_resnet_lfa_NLL": "nanobench_ginio_rd_lfa_nll",
    "ln_resnet_chol_MSE": "nanobench_ginio_rs_chol_mse",
    "ln_resnet_chol_NLL": "nanobench_ginio_rs_chol_nll",
}

# Air-IO experiment dir (relative to airio/, as used by inference_motion.py) -> released folder.
# The two 0.273M "original" VN-AirIO models (Supplement Table 5) are not released (MANIFEST note 5).
AIRIO_RELEASED = {
    "experiments/nanobench_noatt/motion_body_noatt": "nanobench_airio_noatt",
    "experiments/nanobench/motion_body_rot": "nanobench_airio_withatt",
    "experiments/nanobench_noatt_vn_small/motion_body_noatt_vn_small": "nanobench_ginio_a_noatt",
    "experiments/nanobench_rot_vn_small/motion_body_rot_vn_small": "nanobench_ginio_a_withatt",
    "experiments/nanobench_noatt_vn/motion_body_noatt_vn": None,
    "experiments/nanobench_rot_vn/motion_body_rot_vn": None,
}

# TorchScript models of the EqNIO fork: overnight run dir -> released folder (confirmed by the
# `out_dir` recorded in each released parameters.json).
EQNIO_RELEASED = {
    "o2_paper": "tlio_eqnio_o2_ours",            # EqNIO O(2) reference model trained by us (not a paper row)
    "vn_so3_so3aug": "nanobench_ginio_e_so3aug_tlio",      # GINIO-E (TLIO-trained, full-SO(3) augmentation)
    "_existing_vn_so3_t0012": None,              # earlier SO(3) run used by some scripts; not released
}


def insert_paths(*roots):
    """Prepend the given directories to sys.path (same order semantics as the original scripts:
    the last argument ends up first on sys.path)."""
    for p in roots:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def read_test_list(path=TEST_LIST_FILE):
    return [l.strip() for l in Path(path).read_text().split() if l.strip()]


def resolve_ckpt_dir(name, ckpt_root=CKPT_ROOT):
    """Folder of a released checkpoint given its MANIFEST.md name (or an alias). Returns the first
    existing candidate under ckpt_root; if none exists, returns ckpt_root/name (so error messages
    show the canonical path)."""
    root = Path(ckpt_root)
    candidates = [name] + CKPT_ALIASES.get(name, [])
    for alias, als in CKPT_ALIASES.items():          # allow calling with an alias as well
        if name in als:
            candidates += [alias] + [a for a in als if a != name]
    for c in candidates:
        if (root / c).is_dir():
            return root / c
    return root / name


def released_ln_ckpt(old_or_new_name, ckpt_root=CKPT_ROOT):
    """checkpoint_best.pt of a Lie-Neuron model given either the archive name or the released name."""
    name = RELEASED_LN.get(old_or_new_name, old_or_new_name)
    return resolve_ckpt_dir(name, ckpt_root) / "checkpoint_best.pt"


def airio_ckpt_dir(exp_dir, ckpt_root=CKPT_ROOT, airio_root=AIRIO_ROOT):
    """Directory that holds ``best_model.ckpt`` for an Air-IO experiment: the experiment's own
    ``ckpt/`` if present, else the released checkpoint folder, else None."""
    local = Path(airio_root) / exp_dir / "ckpt"
    if (local / "best_model.ckpt").exists():
        return local
    rel = AIRIO_RELEASED.get(str(exp_dir))
    if rel:
        d = resolve_ckpt_dir(rel, ckpt_root)
        if (d / "best_model.ckpt").exists():
            return d
    return None


def ensure_airio_experiment(exp_dir, ckpt_root=CKPT_ROOT, airio_root=AIRIO_ROOT):
    """Make ``airio/<exp_dir>/ckpt/best_model.ckpt`` resolvable for ``inference_motion.py`` by
    symlinking the released checkpoint folder when the experiment directory has no checkpoint.
    Returns the ckpt directory or None if neither exists."""
    local = Path(airio_root) / exp_dir / "ckpt"
    if (local / "best_model.ckpt").exists():
        return local
    rel = AIRIO_RELEASED.get(str(exp_dir))
    if rel is None:
        return None
    src = resolve_ckpt_dir(rel, ckpt_root)
    if not (src / "best_model.ckpt").exists():
        return None
    (Path(airio_root) / exp_dir).mkdir(parents=True, exist_ok=True)
    if not local.exists():
        os.symlink(src.resolve(), local)
    return local
