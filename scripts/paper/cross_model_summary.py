# Paper artefact: TABLE_B_cross_model.csv and fig_eq_vs_noneq_perturbation.png -- the cross-family ID-SO(3) /
# mount-tilt comparison of Supplement Table 10 (all 10 swept models) from the aggregate__*.json files.
"""
Aggregate eq vs non-eq comparison across all Phase B sweep models.

Output CSV: TABLE_B_cross_model.csv
Output figure: fig_eq_vs_noneq.png -- side-by-side ID-SO(3) and calib-mismatch
"""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUTPUT_DIR  # noqa: E402

OUT = Path(OUTPUT_DIR)

MODEL_GROUPS = [
    # (display_name, key, family, color)
    ('AirIO native (no-att)',     'airio_native_noatt', 'non-eq', 'tab:red'),
    ('AirIO native (with-att)',   'airio_native_rot',   'non-eq', 'tab:orange'),
    ('EqNIO O(2)',                'EqNIO_O2',           'eq O(2) [gravity-aligned]', 'tab:brown'),
    ('VN-EqNIO SO(3)',            'VN_EqNIO_SO3',       'eq SO(3)', 'tab:gray'),
    ('VN-AirIO (no-att)',         'vn_airio_noatt',     'eq SO(3) [ckpt drift]', 'tab:olive'),
    ('VN-AirIO (with-att)',       'vn_airio_rot',       'eq SO(3) [ckpt drift]', 'tab:green'),
    ('GINIO-aerial LFA+diag MSE', 'ln_resnet_lfa_MSE',  'eq+LFA', 'tab:blue'),
    ('GINIO-aerial LFA+diag NLL', 'ln_resnet_lfa_NLL',  'eq+LFA', 'tab:cyan'),
    ('GINIO-aerial Spectral MSE', 'ln_resnet_chol_MSE', 'eq SO(3)', 'tab:purple'),
    ('GINIO-aerial Spectral NLL', 'ln_resnet_chol_NLL', 'eq SO(3)', 'tab:pink'),
]

ID_DEGS    = [30, 60, 90, 120, 180]
CALIB_DEGS = [3, 5, 10, 15, 20]


def load_run(m, scenario, seed_tag):
    p = OUT / f'aggregate__{m}__{scenario}__{seed_tag}.json'
    return json.loads(p.read_text()) if p.exists() else None


def collect_deg(m, scenario, deg):
    vals = []
    for s in range(3):
        r = load_run(m, scenario, f'deg{deg}_seed{s}')
        if r: vals.append(r['agg']['rmse_vel'])
    return float(np.mean(vals)) if vals else None


# EqNIO uses 'mount_tilt' scenario key; others use 'calib_mismatch'. Adapter:
def calib_key_for(m):
    return 'mount_tilt' if m in ('EqNIO_O2', 'VN_EqNIO_SO3') else 'calib_mismatch'


def build_csv():
    import csv
    rows = []
    for label, key, family, _ in MODEL_GROUPS:
        clean = load_run(key, 'clean', 'clean')['agg']['rmse_vel']
        row = {'model': label, 'key': key, 'family': family, 'clean': clean}
        ck = calib_key_for(key)
        for deg in ID_DEGS:
            row[f'id_so3_{deg}'] = collect_deg(key, 'id_so3', deg)
        for deg in CALIB_DEGS:
            row[f'calib_{deg}'] = collect_deg(key, ck, deg)
        row['delta_pct_so3_180'] = (row['id_so3_180'] - clean) / clean * 100 if row['id_so3_180'] else None
        row['delta_pct_tilt_20'] = (row['calib_20']   - clean) / clean * 100 if row['calib_20']   else None
        rows.append(row)
    with open(OUT / 'TABLE_B_cross_model.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows


def plot_cross():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: ID-SO(3) curves
    ax = axes[0]
    for label, key, family, color in MODEL_GROUPS:
        clean = load_run(key, 'clean', 'clean')['agg']['rmse_vel']
        xs = [0] + ID_DEGS
        ys = [clean] + [collect_deg(key, 'id_so3', d) for d in ID_DEGS]
        ax.plot(xs, ys, '-o', color=color, label=f'{label} [{family}]', markersize=6, linewidth=1.5)
    ax.set_xlabel('Test-time SO(3) rotation (deg)', fontsize=11)
    ax.set_ylabel('Velocity RMSE (m/s)', fontsize=11)
    ax.set_title('ID-SO(3) perturbation -- equivariance test\n(equivariant = flat, non-equivariant = degrades)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, bbox_to_anchor=(1.0, 1.0))

    # Panel 2: Calib-mismatch curves
    ax = axes[1]
    for label, key, family, color in MODEL_GROUPS:
        clean = load_run(key, 'clean', 'clean')['agg']['rmse_vel']
        ck = calib_key_for(key)
        xs = [0] + CALIB_DEGS
        ys = [clean] + [collect_deg(key, ck, d) for d in CALIB_DEGS]
        ax.plot(xs, ys, '-^', color=color, label=f'{label} [{family}]', markersize=6, linewidth=1.5)
    ax.set_xlabel('Mount-tilt mismatch (deg)', fontsize=11)
    ax.set_ylabel('Velocity RMSE (m/s)', fontsize=11)
    ax.set_title('Calibration mismatch -- mount-tilt robustness\n(LFA + attitude input help; pure-eq is most exposed)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    p = OUT / 'fig_eq_vs_noneq_perturbation.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', p)
    return p


def print_summary(rows):
    fmt_h = '{:30s} | {:10s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>8s}'
    fmt_r = '{:30s} | {:10s} | {:8.4f} | {:8.4f} | {:8.4f} | {:+7.1f}% | {:8.4f} | {:+7.1f}%'
    print(fmt_h.format('Model', 'Family', 'clean', 'so3_30', 'so3_180', 'delta_180', 'tilt_20', 'delta_tilt20'))
    print('-' * 130)
    for r in rows:
        print(fmt_r.format(r['model'][:30], r['family'], r['clean'],
                           r['id_so3_30'], r['id_so3_180'], r['delta_pct_so3_180'],
                           r['calib_20'], r['delta_pct_tilt_20']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default=str(OUTPUT_DIR),
                    help='folder with aggregate__*.json from run_phase_ab*.py; CSV/figure are written here')
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    rows = build_csv()
    print_summary(rows)
    plot_cross()
