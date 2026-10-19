# Third-party code and licenses

GINIO is released under the MIT License (see `LICENSE`). The repository builds on and
redistributes modified copies of the following projects. Their original license terms
continue to apply to the corresponding files.

| Component | Where in this repo | Upstream | License |
|---|---|---|---|
| TLIO (training, data loading, SCEKF tracker) | `src/` (dataloader, network/train.py, test.py, tracker/, utils/), `eqnio/TLIO-master/src/` | https://github.com/CathIAS/TLIO | BSD (Meta Platforms) — see `src/dataloader/LICENSE` |
| EqNIO (O(2)-equivariant canonicalization, frame predictor, rotated-test generators) | `eqnio/TLIO-master/src/network/model_eqvncnn_frame_v2_o2.py`, `dataloader/sequences_dataset.py`, `dataprep_test_augmentation/` | https://github.com/RoyinaJayanth/EqNIO (ICLR 2025) | **No license file published upstream.** Redistributed here with modifications for research reproduction; contact the EqNIO authors before reusing their files outside this purpose. |
| AirIO (recurrent aerial IO: `CodeNetMotion`, datasets, EKF) | `airio/` (model/code.py, datasets/, EKF/, utils/, configs for EuRoC/BlackBird/Pegasus) | https://github.com/Air-IO/Air-IO (IEEE RA-L 2025) | BSD-3-Clause |
| Vector Neurons (VN layers) | `src/models/vn_*.py`, `airio/model/vn_gru.py` (adapted), `eqnio/TLIO-master/src/models/vn_layers.py` | https://github.com/FlyingGiraffe/vnn | MIT |
| Lie Neurons (Lie-algebra layers) | `src/models/lie_neurons_layers.py`, `lie_alg_util.py`, `lie_group_util.py` | https://github.com/UMich-CURLY/LieNeurons | BSD-3-Clause |
| liegroups | dependency (`DiagRotParam` in `src/network/covariance_parametrization.py`) | https://github.com/utiasSTARS/liegroups | MIT |
| PyPose | dependency (`pp.SO3` in `airio/`) | https://github.com/pypose/pypose | Apache-2.0 |

**Checkpoints.** `tlio_eqnio_o2` in the release assets is, to the best of our records, the EqNIO
authors' published TLIO O(2) checkpoint (not trained by us), redistributed unchanged so that the
Table 1 EqNIO row can be reproduced; it remains subject to whatever terms its authors attach, and we
will replace it with a download link on request. Every other released checkpoint was trained by us;
those trained on TLIO golden inherit its CC BY-NC terms.

Datasets are not redistributed in this repository. TLIO golden (CC BY-NC), RIDI, NanoBench
and AquaticVision are distributed by their respective authors under their own terms; see the
README for links.
