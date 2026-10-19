# Paper artefact: TABLE_A_cov_ablation.csv (Supplement Table 9), TABLE_B_id_so3.csv / TABLE_B_calib_mismatch.csv
# (Lie-Neuron rows of Supplement Table 10) and the degradation / coverage figures (Figures 6-7) from the
# aggregate__*.json files written by run_phase_ab.py and the sigma_eq_error*.py results.
"""
Aggregate Phase A+B outputs into paper-ready tables + plots.

Produces (under <out_dir>):
  TABLE_A_cov_ablation.csv      -- head-only: rmse_vel, NLL terms, NEES, coverage, alpha*
  TABLE_A_sigma_eq.csv          -- Sigma-eq / mu-eq error (head-only + LFA-aware)
  TABLE_B_id_so3.csv            -- deg vs rmse_vel per model
  TABLE_B_calib_mismatch.csv    -- deg vs rmse_vel per model
  fig_id_so3.png                -- ATE-vel vs SO(3) degree per model
  fig_calib_mismatch.png        -- ATE-vel vs mount-tilt degree per model
  fig_coverage_curve.png        -- predicted coverage vs ideal (NEES calibration)
"""
import os, sys, json, glob, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUTPUT_DIR  # noqa: E402

OUT = Path(OUTPUT_DIR)

MODELS = ['ln_resnet_lfa_MSE', 'ln_resnet_lfa_NLL', 'ln_resnet_chol_MSE', 'ln_resnet_chol_NLL']
SCENARIOS = ['clean', 'id_so3', 'calib_mismatch']
ID_DEGS    = [0, 30, 60, 90, 120, 180]
CALIB_DEGS = [0, 3, 5, 10, 15, 20]


def load_agg(model, scenario, seed_tag):
    p = OUT / f'aggregate__{model}__{scenario}__{seed_tag}.json'
    if not p.exists(): return None
    return json.loads(p.read_text())


def collect_scenario(model, scenario, degs):
    """For each degree, average over the 3 axis seeds."""
    rows = []
    if scenario == 'clean':
        d = load_agg(model, 'clean', 'clean')
        rows.append({'deg': 0, **d['agg']})
        return rows
    for deg in degs:
        per_seed = []
        for s in range(3):
            d = load_agg(model, scenario, f'deg{deg}_seed{s}')
            if d: per_seed.append(d['agg'])
        if per_seed:
            # average each key
            keys = per_seed[0].keys()
            agg_deg = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
            rows.append({'deg': deg, **agg_deg})
    return rows


def build_table_A():
    """Cov-head head-only table (clean baseline)."""
    import csv
    table = []
    sigma_eq = json.loads((OUT / 'sigma_eq_results.json').read_text())
    sigma_eq_lfa = json.loads((OUT / 'sigma_eq_results_lfa_aware.json').read_text())
    for m in MODELS:
        d = load_agg(m, 'clean', 'clean')['agg']
        head = 'Cholesky' if 'chol' in m else 'Diag'
        lfa  = 'LFA' if 'lfa' in m and 'chol' not in m else 'no-LFA'
        loss = 'NLL' if m.endswith('_NLL') else 'MSE'
        row = dict(
            model=m, head=head, lfa=lfa, loss=loss,
            rmse_vel=d['rmse_vel'],
            mean_NLL=d['mean_nll'],
            mean_logdet=d['mean_logdet'],
            mean_mahal=d['mean_mahal'],
            cov_50=d['coverage_50'],
            cov_70=d['coverage_70'],
            cov_90=d['coverage_90'],
            cov_95=d['coverage_95'],
            sigma_eq_err_naive=sigma_eq[m]['sigma_eq_err_mean'],
            mu_eq_err_naive=sigma_eq[m]['mu_eq_err_mean'],
            sigma_eq_err_lfa_aware=sigma_eq_lfa[m]['sigma_eq_err_mean'],
            mu_eq_err_lfa_aware=sigma_eq_lfa[m]['mu_eq_err_mean'],
        )
        table.append(row)
    with open(OUT / 'TABLE_A_cov_ablation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys()); w.writeheader(); w.writerows(table)
    return table


def build_table_B(scenario, degs):
    """Phase B perturbation table: rows = model, cols = each deg point."""
    import csv
    table = []
    for m in MODELS:
        rows = collect_scenario(m, scenario, degs)
        flat = {'model': m}
        for r in rows:
            d = r['deg']
            flat[f'rmse_vel_{d}'] = r['rmse_vel']
            flat[f'NLL_{d}'] = r['mean_nll']
            flat[f'cov95_{d}'] = r['coverage_95']
        table.append(flat)
    fname = f'TABLE_B_{scenario}.csv'
    if not table: return
    with open(OUT / fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys()); w.writeheader(); w.writerows(table)
    return table


def plot_degradation(scenario, degs, ylabel='RMSE velocity (m/s)', metric='rmse_vel'):
    """Plot metric vs deg per model. Includes clean (deg=0) as anchor."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {'ln_resnet_lfa_MSE':'tab:blue', 'ln_resnet_lfa_NLL':'tab:cyan',
              'ln_resnet_chol_MSE':'tab:orange', 'ln_resnet_chol_NLL':'tab:red'}
    markers = {'MSE':'o', 'NLL':'^'}
    for m in MODELS:
        rows = collect_scenario(m, 'clean', None) + collect_scenario(m, scenario, degs)
        xs = [r['deg'] for r in rows]; ys = [r[metric] for r in rows]
        loss = 'NLL' if m.endswith('_NLL') else 'MSE'
        label = m.replace('ln_resnet_','').replace('_',' ')
        ax.plot(xs, ys, '-' + markers[loss], color=colors[m], label=label, markersize=7)
    ax.set_xlabel('Perturbation magnitude (deg)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{scenario}: {ylabel} vs perturbation magnitude (NanoBench, NN-only)')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    p = OUT / f'fig_{scenario}_{metric}.png'
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def plot_coverage_curve():
    """Per-model coverage at p in {0.5,0.7,0.9,0.95} from clean baseline. Ideal: y=x."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    q_levels = np.array([0.5, 0.7, 0.9, 0.95])
    colors = {'ln_resnet_lfa_MSE':'tab:blue', 'ln_resnet_lfa_NLL':'tab:cyan',
              'ln_resnet_chol_MSE':'tab:orange', 'ln_resnet_chol_NLL':'tab:red'}
    for m in MODELS:
        d = load_agg(m, 'clean', 'clean')['agg']
        emp = [d['coverage_50'], d['coverage_70'], d['coverage_90'], d['coverage_95']]
        ax.plot(q_levels, emp, '-o', color=colors[m], label=m.replace('ln_resnet_',''))
    ax.plot([0,1],[0,1], 'k--', alpha=0.5, label='ideal')
    ax.set_xlabel('Nominal coverage (chi2_3 percentile)')
    ax.set_ylabel('Empirical coverage')
    ax.set_xlim(0.4,1.0); ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title('NEES coverage calibration (NanoBench clean)')
    plt.tight_layout()
    p = OUT / 'fig_coverage_curve.png'
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def print_summary(table_A):
    print('\n=== TABLE A -- Cov-head head-only (NanoBench clean) ===')
    fmt = '{:25s} | head={:8s} | lfa={:6s} | loss={:3s} | rmse_vel={:.4f} | NLL={:7.3f} | cov95={:.3f} | Sigma-eq={:.2e}'
    for r in table_A:
        print(fmt.format(r['model'], r['head'], r['lfa'], r['loss'],
                         r['rmse_vel'], r['mean_NLL'], r['cov_95'], r['sigma_eq_err_naive']))


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default=str(OUTPUT_DIR),
                    help='folder with aggregate__*.json and sigma_eq_results*.json; tables/figures are written here')
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    table_A = build_table_A()
    print_summary(table_A)
    build_table_B('id_so3', ID_DEGS[1:])
    build_table_B('calib_mismatch', CALIB_DEGS[1:])
    plot_degradation('id_so3', ID_DEGS[1:], 'RMSE velocity (m/s)', 'rmse_vel')
    plot_degradation('calib_mismatch', CALIB_DEGS[1:], 'RMSE velocity (m/s)', 'rmse_vel')
    plot_degradation('id_so3', ID_DEGS[1:], 'mean NLL (nats)', 'mean_nll')
    plot_degradation('calib_mismatch', CALIB_DEGS[1:], 'mean NLL (nats)', 'mean_nll')
    plot_coverage_curve()
    print('\nWrote tables + figures to', OUT)


if __name__ == '__main__':
    main()
