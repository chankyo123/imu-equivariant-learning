# Contributing

Issues and pull requests are welcome. For a PR:

1. Keep changes inside one family (`src/`, `airio/`, `eqnio/`) unless the change is shared tooling.
2. Run the CPU equivariance gates before opening the PR:
   `cd airio && python -m model.test_a2_spectral_equivariance` (expects 24/24) and
   `python eqnio/test_vn_eqnio_equivariance.py`.
3. Do not add checkpoints or data to git; they are release assets (see `checkpoints/MANIFEST.md`).
4. If you change a number that the README or MANIFEST quotes, update the MANIFEST first; the README copies it.
