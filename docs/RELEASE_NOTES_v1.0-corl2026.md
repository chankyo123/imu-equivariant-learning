# GINIO v1.0-corl2026 — code and checkpoint release for CoRL 2026

Paper: **GINIO: A Geometric SO(3)-Equivariant Interface for Neural Inertial Odometry** (CoRL 2026, poster) — [OpenReview](https://openreview.net/forum?id=nl8iBTm3Mt) · [arXiv 2609.25338](https://arxiv.org/abs/2609.25338).

## Assets

| asset | contents | size |
|---|---|---:|
| `ginio_checkpoints_tlio.tar.gz` | 8 TLIO checkpoint folders (`tlio_*`) | 432 MB |
| `ginio_checkpoints_ridi.tar.gz` | 6 RIDI checkpoint folders (`ridi_*`) | 346 MB |
| `ginio_checkpoints_nanobench.tar.gz` | 18 NanoBench checkpoint folders (`nanobench_*`) | 409 MB |
| `paper_results.tar.gz` | 709 stored result files behind the "reproduced" values (per-sequence outputs, aggregates, recomputed filter metrics) | 18 MB |
| `SHA256SUMS` | digests of the four archives | — |

`bash scripts/download_checkpoints.sh [tlio|ridi|nanobench|results]` downloads and verifies them into `checkpoints/`.
Each checkpoint archive also carries `INDEX.json` (per-file SHA-256, size, parameter counts, epoch, stored training arguments), the full `SHA256SUMS` of all 32 folders and `RENAMES.txt` (archive directory -> release folder).

## What each checkpoint reproduces

See `checkpoints/MANIFEST.md`: one table per benchmark with the paper row, registry key / loader note, parameter count, paper value vs reproduced value, status (PINNED / CKPT_ONLY / RETRAIN / NOT_PAPER_MODEL / UNSUPPORTED) and 18 traps to read before loading anything. Paper-vs-evidence discrepancies are listed in `docs/KNOWN_ISSUES.md`; the cell-by-cell map is `docs/PAPER_ARTIFACT_MAP.md`.

Not released because nothing survives: the Fetch remounting models (Tables 10–12), AquaticVision (Table 13), GINIO-S on TLIO/RIDI, the RIDI ResNet+SCEKF model, and the generator of the Table 2 ID/SO(3) column for the non-equivariant rows.

## Code

Three instantiations in one repository: `src/` (Lie-Neuron family: GINIO-W/S/RD/RS/R, SCEKF and InEKF trackers), `airio/` (Vector-Neuron AirIO family: GINIO-A), `eqnio/` (canonicalised family: GINIO-E). `scripts/paper/` regenerates the paper's NanoBench tables and figures. Datasets are not redistributed; the exact split lists are in `splits/`.
