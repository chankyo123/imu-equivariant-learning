# Server-B fork patches (provenance only)

These four files are the verbatim `git diff HEAD` of the TLIO-derived experiment forks that
lived on the paper's second machine ("Server B"). They are **not meant to be applied to this
repository**: the GINIO tree already carries the features that matter (see below), and the
forks were throw-away copies of the upstream TLIO repo used for specific filter experiments.
They are shipped so that numbers whose only surviving record is a comment ledger can be traced
back to the exact code that produced them.

| file | base commit (upstream) | what the fork was for |
|---|---|---|
| `TLIO.tracked_changes.patch` | TLIO `d7051c4` | `ATT_ERR_DEG` / `ATT_ERR_AXIS` attitude-error hook in `src/tracker/imu_tracker.py` and the full 3x3 measurement-covariance passthrough in `src/tracker/meas_source_torchscript.py` (needed for the attitude-error sweeps of Fig. 5). Both are **already ported** into `src/tracker/` of this repo (same env-var names, default behaviour unchanged). The third hunk (a `thop` FLOP/param print in `src/network/test.py`) is an ad-hoc profiling hook and was not ported. |
| `TLIO_filter.tracked_changes.patch` | TLIO `d7051c4` | earlier GINIO-W / GINIO-S filter experiments (SCEKF with the equivariant network as measurement source). |
| `covTLIO.tracked_changes.patch` | newTLIO `cc8bf7c` | covariance-head experiments (`*_pca_cov*`, `ln_cov_4regress_*` network files, `json_analysis.py`). |
| `newTLIO.tracked_changes.patch` | newTLIO `cc8bf7c` | earlier filter experiments incl. `covariance_compare.py`. |

The last three forks are the sources of several **ledger-only** entries of Table 1, Table 6 and
Table 7 of the camera-ready (see `PAPER_ARTIFACT_MAP.csv` in the Server-B handoff: status
`LEDGER_ONLY`). Their checkpoints and per-sequence outputs were deleted on Server B; only the
code diffs and the comment ledgers survive.

Base commits refer to the upstream repositories:
`https://github.com/CathIAS/TLIO.git` (TLIO, TLIO_filter) and
`git@github.com:chankyo123/newTLIO.git` (covTLIO, newTLIO).
