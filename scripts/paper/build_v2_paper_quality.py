# Paper artefact: canonical_v2.json (the authoritative per-sequence / aggregate NanoBench numbers behind Table 2),
# table_main.{md,tex} (Table 2 source), table_cov_ablation.md (ATE columns of Supplement Table 9),
# table_lfa_ablation.md and fig1-fig6 (bar / cov-ablation / parameter-efficiency / heatmap / CDF / aggressive).
"""
NanoBench paper-quality results (v2): integrates corrected data from chankyokim-Alienware Archive bundle.

Key corrections:
  - Earlier "ln_resnet+LFA = 0.5805" was actually NON-LFA. Corrected: GINIO-aerial=ln_resnet+LFA
    has ATE_um in 0.79-1.04 range depending on cov head + loss.
  - 4 new LFA variants added (diag/chol × MSE/NLL).
  - ln_resnet no-LFA kept as ABLATION row (clearly labeled).
  - All 8 chankyo models include ATE_meanL2 (TLIO paper convention) alongside ATE_um.

Generates:
  - canonical_v2.json      — unified per-seq + agg for 11 models × 17 seqs (dual metric)
  - table_main.{md,tex,csv}        — Table 1: NanoBench Final (all models)
  - table_cov_ablation.{md,tex}    — Table 2: Cholesky vs diagonal cov head ablation
  - table_lfa_ablation.{md,tex}    — Table 3: LFA vs no-LFA ablation
  - fig1_main_bar.png              — main bar comparison
  - fig2_cov_ablation.png          — Cholesky vs diagonal cov head focused
  - fig3_param_efficiency.png      — params vs ATE
  - fig4_heatmap.png               — per-seq heatmap (aggressive separated)
  - fig5_cdf.png                   — CDF
  - fig6_aggressive_focus.png      — 3-seq aggressive subset detail
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUTPUT_DIR, REFERENCE_INPUTS  # noqa: E402

# Inputs (defaults = the authors' result files shipped in scripts/paper/reference_inputs/):
#   AWS_METRICS  round1_nanobench_metrics.json   (round1_eval.py output: AirIO x2 + original 0.273M VN-AirIO x2)
#   PER_SEQ_CSV  nano_per_seq_dual_metric.csv     (Alienware-side per-sequence ATE_um / ATE_meanL2, 17 seqs x 8 models;
#                                                  formerly Archive/data/)
#   AGG_JSON     nano_aggregate_dual_metric.json  (Alienware-side aggregate RMSE / RPE@2s / drift; formerly Archive/data/)
# All three can be overridden on the command line; OUT defaults to <outputs>/nanobench_final_table.
OUT = str(OUTPUT_DIR / "nanobench_final_table")
AWS_METRICS = str(REFERENCE_INPUTS / "round1_nanobench_metrics.json")
PER_SEQ_CSV = str(REFERENCE_INPUTS / "nano_per_seq_dual_metric.csv")
AGG_JSON = str(REFERENCE_INPUTS / "nano_aggregate_dual_metric.json")

# Test sequences (17)
TEST_SEQS = [
    "B11_random_waypoints_rep1", "B8_star_slow_rep2", "B9_trefoil_medium_rep13",
    "B2_circle_medium_rep2", "B9_trefoil_fast_rep8", "B9_trefoil_slow_rep3",
    "B9_trefoil_medium_rep9", "B2_circle_slow_rep1", "B9_trefoil_slow_rep8",
    "B3_figure8_fast_rep2", "B5_helix_medium_rep1", "B5_helix_rep1",
    "B6_linear_ramp_rep2", "B9_trefoil_slow_rep16", "B10_lissajous_slow_rep1",
    "B2_circle_slow_rep2", "B9_trefoil_medium_rep4",
]

# Aggressive subset (confirmed by chankyo-side: 3 dynamic seqs)
AGGRESSIVE_SEQS = ["B11_random_waypoints_rep1", "B9_trefoil_fast_rep8", "B3_figure8_fast_rep2"]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"],
    "font.size": 12,
    "axes.titlesize": 13.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 260,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.9,
})


def _grid(ax, x=False):
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(False, which="major", axis="both")
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.6)
    if x: ax.grid(True, which="minor", axis="x", linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.6)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.7, color="#aaaaaa", alpha=0.7)
    if x: ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.7, color="#aaaaaa", alpha=0.7)


# ── Build canonical table ───────────────────────────────────────────────────
def load_aws_per_seq():
    """Get per-seq ATE_um from my AWS metrics for the 4 AirIO/VN-AirIO models."""
    with open(AWS_METRICS) as f:
        d = json.load(f)
    out = {}
    for src, lbl in [
        ("AirIO no-att (baseline)", "AirIO_no_att"),
        ("AirIO with-att (baseline)", "AirIO_with_att"),
        ("VN no-att", "VN_AirIO_no_att"),
        ("VN with-att", "VN_AirIO_with_att"),
    ]:
        if src not in d: continue
        out[lbl] = {seq: {"ATE_um": v["ATE_aligned"],
                          "RMSE": v["RMSE_no_align"],
                          "RPE2s": v["RPE@2s"],
                          "drift": v["drift_endpoint"]} for seq, v in d[src].items()}
    return out


def load_archive_per_seq():
    """Load per-seq dual metric CSV from Archive."""
    out = {}
    with open(PER_SEQ_CSV) as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    # Detect model column prefixes (excluding 'seq')
    cols = rows[0].keys() if rows else []
    model_names = sorted(set(c.rsplit("_ATE_", 1)[0] for c in cols if c != "seq"))
    for m in model_names:
        out[m] = {}
        for r in rows:
            seq = r["seq"]
            ate_um_key = f"{m}_ATE_um"
            ate_l2_key = f"{m}_ATE_meanL2"
            if ate_um_key in r:
                try:
                    out[m][seq] = {"ATE_um": float(r[ate_um_key]),
                                   "ATE_meanL2": float(r[ate_l2_key])}
                except (ValueError, TypeError):
                    pass
    return out


def load_archive_agg():
    with open(AGG_JSON) as f:
        return json.load(f)


# ── Model registry (paper-display order + metadata) ────────────────────────
MODELS = [
    # Non-equivariant baselines
    {"key": "AirIO",
     "label": "AirIO (with-att, native)",
     "category": "non-eq baseline", "frame": "body+rot",
     "params_M": 0.39, "cov_head": "diag",
     "loss": "Huber+rotloss", "lfa": False, "source": "chankyo"},
    {"key": "AirIO_no_att",
     "label": "AirIO (no-att, native)",
     "category": "non-eq baseline", "frame": "body",
     "params_M": 0.39, "cov_head": "diag",
     "loss": "Huber+rotloss", "lfa": False, "source": "aws"},
    {"key": "AirIO_with_att",
     "label": "AirIO with-att (AWS-trained)",
     "category": "non-eq baseline", "frame": "body+rot",
     "params_M": 0.39, "cov_head": "diag",
     "loss": "Huber+rotloss", "lfa": False, "source": "aws",
     "note": "cross-check of chankyo's AirIO row"},
    {"key": "ResNet",
     "label": "ResNet1D (non-eq, big baseline)",
     "category": "non-eq baseline", "frame": "body",
     "params_M": 5.43, "cov_head": "diag",
     "loss": "MSE", "lfa": False, "source": "chankyo"},
    # Equivariant / canonical-frame
    {"key": "EqNIO_O2",
     "label": "EqNIO O(2)",
     "category": "equivariant (yaw)", "frame": "gravity-aligned",
     "params_M": 6.03, "cov_head": "spectral (2D in eq frame)",
     "loss": "Huber+NLL", "lfa": False, "source": "chankyo"},
    # ln_resnet ABLATION (no LFA — actually best i.i.d. but no equivariance prior)
    {"key": "ln_resnet_nolfa_t119_ablation_only",
     "label": "ln_resnet (no LFA, ablation)",
     "category": "ablation", "frame": "body",
     "params_M": 0.57, "cov_head": "diag",
     "loss": "NLL", "lfa": False, "source": "chankyo",
     "note": "honest ablation — LFA preprocessing disabled"},
    # GINIO-aerial variants (the actual main contribution)
    {"key": "ln_resnet_LFA_MSE",
     "label": "GINIO-aerial (LFA, diag cov, MSE)",
     "category": "GINIO-aerial", "frame": "body + LFA",
     "params_M": 0.57, "cov_head": "diagonal",
     "loss": "MSE", "lfa": True, "source": "chankyo"},
    {"key": "ln_resnet_LFA_NLL",
     "label": "GINIO-aerial (LFA, diag cov, NLL)",
     "category": "GINIO-aerial", "frame": "body + LFA",
     "params_M": 0.57, "cov_head": "diagonal",
     "loss": "NLL", "lfa": True, "source": "chankyo"},
    {"key": "ln_resnet_chol_LFA_MSE",
     "label": "GINIO-aerial (LFA, Cholesky cov, MSE)",
     "category": "GINIO-aerial (Σ-eq)", "frame": "body + LFA",
     "params_M": 0.80, "cov_head": "Cholesky (3-param)",
     "loss": "MSE", "lfa": True, "source": "chankyo"},
    {"key": "ln_resnet_chol_LFA_NLL",
     "label": "GINIO-aerial (LFA, Cholesky cov, NLL)",
     "category": "GINIO-aerial (Σ-eq)", "frame": "body + LFA",
     "params_M": 0.80, "cov_head": "Cholesky (3-param)",
     "loss": "NLL", "lfa": True, "source": "chankyo"},
    # GINIO-AirIO (our VN-AirIO)
    {"key": "VN_AirIO_with_att",
     "label": "GINIO-AirIO (VN-AirIO, with-att)",
     "category": "GINIO-AirIO", "frame": "body+rot",
     "params_M": 0.40, "cov_head": "diag-from-norms",
     "loss": "Huber+rotloss", "lfa": False, "source": "aws"},
    {"key": "VN_AirIO_no_att",
     "label": "GINIO-AirIO (VN-AirIO, no-att)",
     "category": "GINIO-AirIO", "frame": "body",
     "params_M": 0.40, "cov_head": "diag-from-norms",
     "loss": "Huber+rotloss", "lfa": False, "source": "aws"},
]


def agg(per_seq, seqs, key="ATE_um"):
    vals = []
    for s in seqs:
        if s in per_seq and key in per_seq[s]:
            v = per_seq[s][key]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
    if not vals: return {"n": 0, "mean": None, "median": None}
    return {"n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals))}


def build_canonical():
    arch_per_seq = load_archive_per_seq()
    aws_per_seq = load_aws_per_seq()
    arch_agg = load_archive_agg()
    canonical = {}
    for m in MODELS:
        k = m["key"]
        per_seq = arch_per_seq.get(k, aws_per_seq.get(k, {}))
        # Get RMSE/RPE/drift from agg (chankyo data only has these at agg level)
        agg_rec = arch_agg.get(k, {})
        canonical[k] = {
            **m,
            "per_seq": per_seq,
            "ATE_um_stats":   agg(per_seq, TEST_SEQS, "ATE_um"),
            "ATE_meanL2_stats": agg(per_seq, TEST_SEQS, "ATE_meanL2"),
            "ATE_um_aggressive":   agg(per_seq, AGGRESSIVE_SEQS, "ATE_um"),
            "ATE_meanL2_aggressive": agg(per_seq, AGGRESSIVE_SEQS, "ATE_meanL2"),
            "RMSE_mean": agg_rec.get("RMSE_no_align_mean"),
            "RPE2s_mean": agg_rec.get("RPE_2s_mean"),
            "drift_mean": agg_rec.get("drift_endpoint_mean"),
        }
        # For AWS-only models, compute RMSE/RPE/drift directly
        if not canonical[k]["RMSE_mean"]:
            rmse_vals = [per_seq[s].get("RMSE") for s in per_seq if "RMSE" in per_seq[s]]
            rpe_vals = [per_seq[s].get("RPE2s") for s in per_seq if "RPE2s" in per_seq[s] and not np.isnan(per_seq[s]["RPE2s"])]
            drift_vals = [per_seq[s].get("drift") for s in per_seq if "drift" in per_seq[s]]
            if rmse_vals: canonical[k]["RMSE_mean"] = float(np.mean(rmse_vals))
            if rpe_vals: canonical[k]["RPE2s_mean"] = float(np.mean(rpe_vals))
            if drift_vals: canonical[k]["drift_mean"] = float(np.mean(drift_vals))
    return canonical


# ── Tables ──────────────────────────────────────────────────────────────────
def write_main_table_md(c):
    lines = [
        "# Table 1 — NanoBench Final Table (Standard, NN-only)",
        "",
        "n=17 test sequences. Lower is better.",
        "ATE_um = Umeyama-aligned RMSE (our default), ATE_meanL2 = raw mean L2 (TLIO paper convention).",
        "",
        "| # | Model | Frame | Params (M) | Cov head | Σ-eq? | Loss | ATE_um↓ | ATE_med | ATE_meanL2↓ | RMSE | RPE@2s | drift |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    sig_eq_lookup = {
        "diagonal":              "✗ (≈2e-3)",
        "diag":                  "✗ (≈2e-3)",
        "diag-from-norms":       "✗ (≈2e-3)",
        "spectral (2D in eq frame)": "≈ (frame-cond.)",
        "Cholesky (3-param)":    "✓ (≤1e-12)",
    }
    for i, m in enumerate(MODELS, 1):
        rec = c[m["key"]]
        s_um = rec["ATE_um_stats"]
        s_l2 = rec["ATE_meanL2_stats"]
        if s_um["mean"] is None: continue
        ate_um   = f"{s_um['mean']:.3f}"
        ate_med  = f"{s_um['median']:.3f}"
        ate_l2   = f"{s_l2['mean']:.3f}" if s_l2["mean"] is not None else "—"
        rmse = f"{rec['RMSE_mean']:.3f}" if rec.get('RMSE_mean') else "—"
        rpe  = f"{rec['RPE2s_mean']:.3f}" if rec.get('RPE2s_mean') else "—"
        dr   = f"{rec['drift_mean']:.3f}" if rec.get('drift_mean') else "—"
        sig  = sig_eq_lookup.get(m["cov_head"], "?")
        lines.append(
            f"| {i} | {m['label']} | {m['frame']} | {m['params_M']:.2f} | "
            f"{m['cov_head']} | {sig} | {m['loss']} | "
            f"{ate_um} | {ate_med} | {ate_l2} | {rmse} | {rpe} | {dr} |"
        )
    lines.append("")
    lines.append("## Aggressive subset (n=3 dynamic seqs: random_waypoints, trefoil_fast, figure8_fast)")
    lines.append("")
    lines.append("| Model | ATE_um (mean) | ATE_meanL2 (mean) |")
    lines.append("|---|---:|---:|")
    for m in MODELS:
        rec = c[m["key"]]
        a_um = rec["ATE_um_aggressive"]
        a_l2 = rec["ATE_meanL2_aggressive"]
        if a_um["mean"] is None: continue
        l2 = f"{a_l2['mean']:.3f}" if a_l2["mean"] is not None else "—"
        lines.append(f"| {m['label']} | {a_um['mean']:.3f} | {l2} |")
    with open(f"{OUT}/table_main.md", "w") as f:
        f.write("\n".join(lines))


def write_cov_ablation_md(c):
    """Cov head ablation within ln_resnet+LFA backbone (same arch, same data)."""
    pairs = [
        ("MSE", "ln_resnet_LFA_MSE", "ln_resnet_chol_LFA_MSE"),
        ("NLL", "ln_resnet_LFA_NLL", "ln_resnet_chol_LFA_NLL"),
    ]
    lines = [
        "# Table 2 — Covariance head ablation (within ln_resnet+LFA backbone)",
        "",
        "Same architecture, same training data, same loss. Only the covariance head is swapped.",
        "Σ-eq? = covariance transforms as a (1,1) tensor under SO(3): Σ(ρ(R)Z) = R Σ(Z) Rᵀ.",
        "",
        "| Loss | Cov head | Σ-eq? | Σ-eq err | Params (M) | ATE_um ↓ | ATE_meanL2 ↓ | Δ vs diag |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for loss, dk, ck in pairs:
        d = c[dk]
        ch = c[ck]
        d_um = d["ATE_um_stats"]["mean"]
        c_um = ch["ATE_um_stats"]["mean"]
        d_l2 = d["ATE_meanL2_stats"]["mean"]
        c_l2 = ch["ATE_meanL2_stats"]["mean"]
        delta = (c_um - d_um) / d_um * 100
        lines.append(
            f"| {loss} | Diagonal | ✗ | ~2e-3 | 0.57 | "
            f"{d_um:.3f} | {d_l2:.3f} | — |"
        )
        lines.append(
            f"| {loss} | **Cholesky (3-param)** | ✓ | ~1e-12 | 0.80 | "
            f"**{c_um:.3f}** | **{c_l2:.3f}** | **{delta:+.1f}%** |"
        )
    lines.append("")
    lines.append("**Finding**: SO(3)-equivariant Cholesky cov head improves ATE by **13.2% (MSE) and 19.8% (NLL)** "
                 "over the non-equivariant diagonal cov head on NanoBench aerial dynamics. This is the largest "
                 "controllable factor we found for i.i.d. accuracy on this domain.")
    with open(f"{OUT}/table_cov_ablation.md", "w") as f:
        f.write("\n".join(lines))


def write_lfa_ablation_md(c):
    """LFA vs no-LFA, within ln_resnet backbone, NLL loss."""
    lines = [
        "# Table 3 — LFA preprocessing ablation (within ln_resnet, NLL, diagonal cov)",
        "",
        "Same architecture (ln_resnet, 0.57M), same loss (NLL), same diagonal cov head.",
        "Only difference: LFA last-frame anchor preprocessing applied to IMU input.",
        "",
        "| Variant | LFA? | ATE_um ↓ | ATE_meanL2 ↓ | Note |",
        "|---|---|---:|---:|---|",
    ]
    no_lfa = c["ln_resnet_nolfa_t119_ablation_only"]["ATE_um_stats"]["mean"]
    lfa = c["ln_resnet_LFA_NLL"]["ATE_um_stats"]["mean"]
    no_lfa_l2 = c["ln_resnet_nolfa_t119_ablation_only"]["ATE_meanL2_stats"]["mean"]
    lfa_l2 = c["ln_resnet_LFA_NLL"]["ATE_meanL2_stats"]["mean"]
    delta = (lfa - no_lfa) / no_lfa * 100
    lines.append(f"| ln_resnet (no LFA) | ✗ | {no_lfa:.3f} | {no_lfa_l2:.3f} | i.i.d.-best on NanoBench |")
    lines.append(f"| GINIO-aerial (ln_resnet + LFA) | ✓ | {lfa:.3f} | {lfa_l2:.3f} | LFA prior; ATE_um {delta:+.1f}% vs no-LFA |")
    lines.append("")
    lines.append("**Finding**: LFA preprocessing HURTS i.i.d. ATE by +80% on NanoBench. This is an HONEST ablation. "
                 "The contribution of LFA must therefore rest on **robustness under domain shift** "
                 "(ID-SO(3) perturbation, calibration mismatch), not i.i.d. accuracy.")
    lines.append("")
    lines.append("> Per Wang et al. 2023 (\"Surprising Effectiveness of Equivariant Models in Domains with Latent Symmetry\"), "
                 "equivariant priors trade i.i.d. fit for invariance to nuisance transforms. The Phase-2 perturbation evaluation "
                 "(forthcoming) is required to validate the LFA design choice on NanoBench aerial.")
    with open(f"{OUT}/table_lfa_ablation.md", "w") as f:
        f.write("\n".join(lines))


def write_main_latex(c):
    rows = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{NanoBench Final Table --- Standard scenario (NN only, n=17 test sequences). "
        r"Lower is better. ATE$_{um}$ = Umeyama-aligned RMSE; ATE$_{L_2}$ = raw mean L2 (TLIO convention). "
        r"$\Sigma$-eq.\ indicates whether the covariance head transforms as a rank-2 tensor under $\mathrm{SO}(3)$. "
        r"\textbf{Bold} = best within category; \underline{underline} = best overall on metric.}",
        r"\label{tab:nanobench_final}",
        r"\begin{tabular}{llrcclrrrrr}",
        r"\toprule",
        r"Family & Model & Params(M) & Cov head & $\Sigma$-eq.\ & Loss & ATE$_{um}\downarrow$ & ATE$_{L_2}\downarrow$ & RMSE & RPE@2s & drift \\",
        r"\midrule",
    ]
    sig_eq_lookup = {
        "diagonal":              r"\xmark",
        "diag":                  r"\xmark",
        "diag-from-norms":       r"\xmark",
        "spectral (2D in eq frame)": r"\cmark*",
        "Cholesky (3-param)":    r"\cmark",
    }
    # Find best per metric
    ate_um_vals = [(m["key"], c[m["key"]]["ATE_um_stats"]["mean"]) for m in MODELS if c[m["key"]]["ATE_um_stats"]["mean"] is not None]
    best_ate_um = min(ate_um_vals, key=lambda x: x[1])[0]
    last_cat = None
    for m in MODELS:
        rec = c[m["key"]]
        s = rec["ATE_um_stats"]
        if s["mean"] is None: continue
        if m["category"] != last_cat:
            if last_cat is not None: rows.append(r"\midrule")
            last_cat = m["category"]
        ate_um = f"{s['mean']:.3f}"
        if m["key"] == best_ate_um:
            ate_um = r"\underline{" + ate_um + "}"
        ate_l2 = rec["ATE_meanL2_stats"]["mean"]
        ate_l2_s = f"{ate_l2:.3f}" if ate_l2 is not None else "---"
        rmse = f"{rec['RMSE_mean']:.3f}" if rec.get('RMSE_mean') else "---"
        rpe  = f"{rec['RPE2s_mean']:.3f}" if rec.get('RPE2s_mean') else "---"
        dr   = f"{rec['drift_mean']:.3f}" if rec.get('drift_mean') else "---"
        sig = sig_eq_lookup.get(m["cov_head"], "?")
        rows.append(
            f"{m['category']} & {m['label']} & {m['params_M']:.2f} & {m['cov_head']} & {sig} & {m['loss']} & "
            f"{ate_um} & {ate_l2_s} & {rmse} & {rpe} & {dr} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table*}")
    with open(f"{OUT}/table_main.tex", "w") as f:
        f.write("\n".join(rows))


# ── Figures ──────────────────────────────────────────────────────────────────
COLORS = {
    "non-eq baseline":   "#9e9e9e",
    "equivariant (yaw)": "#fb9a99",
    "ablation":          "#cccc88",
    "GINIO-aerial":      "#a6cee3",
    "GINIO-aerial (Σ-eq)": "#1f78b4",
    "GINIO-AirIO":       "#6a3d9a",
}


def fig_main_bar(c):
    """Horizontal bar plot, all models, colored by category."""
    rows = [m for m in MODELS if c[m["key"]]["ATE_um_stats"]["mean"] is not None]
    rows = list(reversed(rows))  # plot top-down with best at top
    labels = [m["label"] for m in rows]
    ate_um = [c[m["key"]]["ATE_um_stats"]["mean"] for m in rows]
    ate_l2 = [c[m["key"]]["ATE_meanL2_stats"]["mean"] for m in rows]
    cats = [m["category"] for m in rows]
    colors = [COLORS.get(cat, "#888888") for cat in cats]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    y = np.arange(len(labels))

    # Left: ATE_um
    ax = axes[0]
    bars = ax.barh(y, ate_um, color=colors, edgecolor="black", linewidth=0.6)
    for yi, v in zip(y, ate_um):
        ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ATE$_{um}$ (m) ↓")
    ax.set_title("(a) Umeyama-aligned ATE")
    ax.axvline(min(ate_um), color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    _grid(ax, x=True)

    # Right: ATE_meanL2
    ax = axes[1]
    valid = [(yi, v) for yi, v in zip(y, ate_l2) if v is not None]
    if valid:
        y_v = [x[0] for x in valid]
        ate_l2_v = [x[1] for x in valid]
        ax.barh(y_v, ate_l2_v, color=[colors[i] for i in y_v], edgecolor="black", linewidth=0.6)
        for yi, v in zip(y_v, ate_l2_v):
            ax.text(v + 0.04, yi, f"{v:.3f}", va="center", fontsize=9.5)
        ax.set_xlabel("ATE$_{meanL_2}$ (m) ↓")
        ax.set_title("(b) Raw mean L2 ATE (TLIO convention)")
        ax.axvline(min(ate_l2_v), color="black", linewidth=0.6, linestyle=":", alpha=0.5)
        _grid(ax, x=True)

    # Legend
    handles = [Patch(facecolor=col, edgecolor="black", label=cat) for cat, col in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True,
               edgecolor="black", framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("NanoBench Final — Standard scenario (NN-only, n=17 test sequences)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    out = f"{OUT}/fig1_main_bar.png"
    plt.savefig(out)
    plt.close()
    return out


def fig_cov_ablation(c):
    """Focused 2×2 cov ablation plot: Cholesky vs diagonal, MSE vs NLL."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    pairs = [
        ("MSE", "ln_resnet_LFA_MSE", "ln_resnet_chol_LFA_MSE"),
        ("NLL", "ln_resnet_LFA_NLL", "ln_resnet_chol_LFA_NLL"),
    ]
    for ax, metric in zip(axes, ["ATE_um_stats", "ATE_meanL2_stats"]):
        x = np.arange(2)
        w = 0.36
        diag_vals = [c[p[1]][metric]["mean"] for p in pairs]
        chol_vals = [c[p[2]][metric]["mean"] for p in pairs]
        b1 = ax.bar(x - w/2, diag_vals, w, color="#C44E52", edgecolor="black", linewidth=0.7,
                    label="Diagonal cov (non-eq, ε_Σ ≈ 2e-3)")
        b2 = ax.bar(x + w/2, chol_vals, w, color="#4C72B0", edgecolor="black", linewidth=0.7,
                    label="Cholesky cov (SO(3)-eq, ε_Σ ≤ 1e-12)")
        for xi, v in zip(x - w/2, diag_vals):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5)
        for xi, v in zip(x + w/2, chol_vals):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5)
        # Delta annotations
        for i, (d_v, c_v) in enumerate(zip(diag_vals, chol_vals)):
            delta = (c_v - d_v) / d_v * 100
            ax.annotate(f"Δ {delta:+.1f}%", (i, max(d_v, c_v) * 1.10),
                        ha="center", fontsize=10, color="#2a7a2a", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p[0]} training" for p in pairs])
        name = "ATE_um" if metric == "ATE_um_stats" else "ATE_meanL2"
        ax.set_ylabel(f"{name} (m, n=17) ↓")
        ax.set_title(f"({chr(97 + (0 if metric == 'ATE_um_stats' else 1))}) {name}")
        ax.legend(loc="upper left", frameon=True, edgecolor="black", framealpha=0.92, fontsize=9)
        _grid(ax)
    fig.suptitle("Covariance-head ablation within GINIO-aerial backbone (NanoBench)",
                 fontsize=13.5, y=1.02)
    plt.tight_layout()
    out = f"{OUT}/fig2_cov_ablation.png"
    plt.savefig(out)
    plt.close()
    return out


def fig_param_efficiency(c):
    rows = [m for m in MODELS if c[m["key"]]["ATE_um_stats"]["mean"] is not None]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for m in rows:
        p = m["params_M"]
        a = c[m["key"]]["ATE_um_stats"]["mean"]
        cat = m["category"]
        ax.scatter(p, a, color=COLORS.get(cat, "#888"), s=180, edgecolor="black",
                   linewidth=1.0, zorder=3)
        # Compact label
        label = m["label"]
        # Shorten
        if "GINIO-aerial" in label and "Cholesky" in label: label = "GINIO-aerial+Chol"
        elif "GINIO-aerial" in label: label = "GINIO-aerial+diag"
        elif "GINIO-AirIO" in label: label = "GINIO-AirIO"
        elif "ln_resnet (no LFA" in label: label = "ln_resnet (no LFA)"
        ax.annotate(label, (p, a), xytext=(8, 6), textcoords="offset points",
                    fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("ATE$_{um}$ (m) ↓")
    ax.set_title("Parameter efficiency on NanoBench (Standard)")
    handles = [Patch(facecolor=col, edgecolor="black", label=cat) for cat, col in COLORS.items()]
    ax.legend(handles=handles, loc="best", frameon=True, edgecolor="black", framealpha=0.92, fontsize=9)
    _grid(ax, x=True)
    out = f"{OUT}/fig3_param_efficiency.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def fig_heatmap(c):
    """Per-seq ATE_um heatmap with aggressive separator."""
    rows = [m for m in MODELS if c[m["key"]]["ATE_um_stats"]["mean"] is not None]
    sorted_seqs = AGGRESSIVE_SEQS + [s for s in TEST_SEQS if s not in AGGRESSIVE_SEQS]
    mat = np.full((len(rows), len(sorted_seqs)), np.nan)
    for i, m in enumerate(rows):
        per = c[m["key"]]["per_seq"]
        for j, s in enumerate(sorted_seqs):
            v = per.get(s, {}).get("ATE_um")
            if v is not None: mat[i, j] = v

    fig, ax = plt.subplots(figsize=(15, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0.2, vmax=3.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([m["label"] for m in rows], fontsize=9.5)
    ax.set_xticks(range(len(sorted_seqs)))
    ax.set_xticklabels([s.replace("rep", "r") for s in sorted_seqs], rotation=70, ha="right", fontsize=8)
    for i in range(len(rows)):
        for j in range(len(sorted_seqs)):
            if not np.isnan(mat[i, j]):
                col = "white" if mat[i, j] > 2.0 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=col)
    sep_x = len(AGGRESSIVE_SEQS) - 0.5
    ax.axvline(sep_x, color="black", linewidth=2)
    ax.text(len(AGGRESSIVE_SEQS) / 2 - 0.5, -1.1, f"Aggressive (n={len(AGGRESSIVE_SEQS)})",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(len(AGGRESSIVE_SEQS) + (len(TEST_SEQS) - len(AGGRESSIVE_SEQS)) / 2 - 0.5, -1.1,
            f"Standard \\ aggressive (n={len(TEST_SEQS) - len(AGGRESSIVE_SEQS)})",
            ha="center", fontsize=11, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("ATE$_{um}$ (m)")
    ax.set_title("Per-sequence ATE$_{um}$ — NanoBench Standard")
    out = f"{OUT}/fig4_heatmap.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def fig_cdf(c):
    rows = [m for m in MODELS if c[m["key"]]["ATE_um_stats"]["mean"] is not None]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for m in rows:
        per = c[m["key"]]["per_seq"]
        vals = sorted([v["ATE_um"] for v in per.values() if "ATE_um" in v])
        n = len(vals)
        cdf = np.arange(1, n + 1) / n * 100
        ax.plot(vals, cdf, color=COLORS.get(m["category"], "#666"), linewidth=2.0,
                marker="o", markersize=4.5, markeredgecolor="black", markeredgewidth=0.3,
                label=m["label"], alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlabel("ATE$_{um}$ (m, log) ↓")
    ax.set_ylabel("Fraction of test sequences (%)")
    ax.set_title("CDF of per-sequence ATE$_{um}$ — NanoBench Standard")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, edgecolor="black", framealpha=0.92)
    ax.set_ylim(0, 105)
    _grid(ax, x=True)
    out = f"{OUT}/fig5_cdf.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def fig_aggressive(c):
    rows = [m for m in MODELS if c[m["key"]]["ATE_um_aggressive"]["mean"] is not None]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(rows)); w = 0.38
    um_vals = [c[m["key"]]["ATE_um_aggressive"]["mean"] for m in rows]
    l2_vals = [c[m["key"]]["ATE_meanL2_aggressive"]["mean"] for m in rows]
    colors = [COLORS.get(m["category"], "#666") for m in rows]
    ax.bar(x - w/2, um_vals, w, color=colors, edgecolor="black", linewidth=0.6,
           label=r"ATE$_{um}$", alpha=0.95)
    ax.bar(x + w/2, [v if v is not None else 0 for v in l2_vals], w, color=colors,
           edgecolor="black", linewidth=0.6, hatch="//", label=r"ATE$_{meanL_2}$", alpha=0.65)
    ax.set_xticks(x)
    short_labels = []
    for m in rows:
        lbl = m["label"]
        if "GINIO-aerial" in lbl and "Cholesky" in lbl: lbl = "GINIO-aerial\n+Chol\n(" + m["loss"] + ")"
        elif "GINIO-aerial" in lbl: lbl = "GINIO-aerial\n+diag\n(" + m["loss"] + ")"
        elif "GINIO-AirIO" in lbl: lbl = "GINIO-AirIO\n" + ("with-att" if "with-att" in lbl else "no-att")
        elif "AirIO" in lbl: lbl = "AirIO\n" + ("with-att" if "with-att" in lbl else "no-att")
        elif "no LFA" in lbl: lbl = "ln_resnet\n(no LFA)"
        elif "EqNIO" in lbl: lbl = "EqNIO\nO(2)"
        elif "ResNet1D" in lbl: lbl = "ResNet1D\n(non-eq)"
        short_labels.append(lbl)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylabel("ATE (m, n=3 aggressive seqs)")
    ax.set_title(f"Aggressive subset (n=3: random_waypoints, trefoil_fast, figure8_fast)")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=0.92)
    _grid(ax)
    out = f"{OUT}/fig6_aggressive.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def main():
    global OUT, AWS_METRICS, PER_SEQ_CSV, AGG_JSON
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aws_metrics", default=AWS_METRICS,
                    help="round1_eval.py metrics JSON (AirIO / VN-AirIO per-sequence metrics)")
    ap.add_argument("--per_seq_csv", default=PER_SEQ_CSV,
                    help="per-sequence ATE_um / ATE_meanL2 CSV for the Lie-Neuron, ResNet1D, EqNIO O(2) and AirIO rows")
    ap.add_argument("--agg_json", default=AGG_JSON,
                    help="aggregate RMSE / RPE@2s / drift JSON for the same rows")
    ap.add_argument("--out_dir", default=OUT, help="where canonical_v2.json, tables and figures are written")
    args = ap.parse_args()
    OUT, AWS_METRICS, PER_SEQ_CSV, AGG_JSON = args.out_dir, args.aws_metrics, args.per_seq_csv, args.agg_json
    os.makedirs(OUT, exist_ok=True)
    c = build_canonical()
    # Save canonical
    serializable = {}
    for k, v in c.items():
        sv = {kk: vv for kk, vv in v.items()}
        serializable[k] = sv
    with open(f"{OUT}/canonical_v2.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    # Tables
    write_main_table_md(c)
    write_cov_ablation_md(c)
    write_lfa_ablation_md(c)
    write_main_latex(c)
    # Figures
    figs = [fig_main_bar(c), fig_cov_ablation(c), fig_param_efficiency(c),
            fig_heatmap(c), fig_cdf(c), fig_aggressive(c)]
    print("=== Paper-quality NanoBench artifacts (v2) ===")
    print(f"  canonical_v2.json")
    print(f"  table_main.md / .tex")
    print(f"  table_cov_ablation.md")
    print(f"  table_lfa_ablation.md")
    for f in figs:
        print(f"  {os.path.basename(f)}")
    print()
    # Print summary
    print("Standard scenario summary:")
    print(f"  {'Model':<45s} {'M':>5s} {'ATE_um':>8s} {'ATE_L2':>8s}")
    print("  " + "-" * 73)
    for m in MODELS:
        rec = c[m["key"]]
        s = rec["ATE_um_stats"]
        l = rec["ATE_meanL2_stats"]
        if s["mean"] is None: continue
        l2_s = f"{l['mean']:.3f}" if l['mean'] is not None else "  ---"
        print(f"  {m['label']:<45s} {m['params_M']:5.2f} {s['mean']:8.3f} {l2_s:>8s}")


if __name__ == "__main__":
    main()
