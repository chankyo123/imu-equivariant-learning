"""
LFA (Last-Frame Alignment) preprocessing — extracted standalone from
imu-equivariant-learning/src/dataloader/sequences_dataset.py:421-456.

GINIO-aerial uses this LFA step on body-frame IMU windows BEFORE feeding the
network. The math:

Given a window of T IMU samples with per-step ground-truth orientation
R_t_W in SO(3) (sensor-in-world rotation), LFA computes:

    R_last_W = R_{T-1}_W                     # orientation at the LAST sample
    R_t_last = R_last_W^T @ R_t_W            # orientation at time t expressed
                                             # in LAST-frame body axes
    imu_t_last = R_t_last @ imu_t_World      # rotate IMU into last-frame body axes

By construction, R_{T-1}_last = I, so the last sample is unchanged. Earlier
samples are rotated to express what the IMU would have measured had the
last-frame body axes been fixed throughout the window.

Targets (velocity / displacement) are NOT rotated by LFA — they remain in
world frame.

Gate flag used in the original codebase:
    align_last_imu = ("nolfa" not in data_path)   # default ON
Train a non-LFA baseline by including "nolfa" in the dataset directory name.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def apply_lfa(imu_window, rot_quat_window):
    """
    Last-Frame Alignment preprocessing for a single IMU window.

    Args:
        imu_window: (6, T) numpy array. Rows 0..2 = gyro (rad/s), 3..5 = accel (m/s^2)
                    in some fixed frame (typically world or world-yaw-aligned).
        rot_quat_window: (T, 4) numpy array of sensor-in-world quaternions
                         (xyzw convention, matching scipy.spatial.transform.Rotation).

    Returns:
        (6, T) numpy array — IMU rotated into the LAST-frame body axes.
    """
    assert imu_window.ndim == 2 and imu_window.shape[0] == 6, \
        f"Expected (6, T) IMU, got {imu_window.shape}"
    T = imu_window.shape[1]
    assert rot_quat_window.shape == (T, 4), \
        f"Expected ({T}, 4) quaternions, got {rot_quat_window.shape}"

    # R_last_W (3, 3): sensor-in-world rotation at LAST window sample
    R_last_W = Rotation.from_quat(rot_quat_window[-1:]).as_matrix().transpose(0, 2, 1)
    R_last_W = np.squeeze(R_last_W, axis=0)  # (3, 3) = R_last_W^T (note transpose above)

    # R_t_W (T, 3, 3): sensor-in-world for every sample
    R_n_W = Rotation.from_quat(rot_quat_window).as_matrix()

    # R_t_last (T, 3, 3): orientation in last-frame body axes
    R_n_last = np.einsum('ij,tjk->tik', R_last_W, R_n_W)

    # Rotate gyro and accel into last-frame axes
    out = imu_window.copy()
    out[:3] = np.einsum("tij,jt->it", R_n_last, imu_window[:3])
    out[3:6] = np.einsum("tij,jt->it", R_n_last, imu_window[3:6])
    return out


# ---------------------------------------------------------------------------
# Smoke test (run as `python lfa_preprocessing.py` to verify last-frame anchor)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    T = 200
    rng = np.random.default_rng(0)
    # Random IMU + tumbling rotations
    imu = rng.standard_normal((6, T)).astype(np.float64)
    rotvecs = rng.standard_normal((T, 3)) * 0.1
    quats = Rotation.from_rotvec(rotvecs).as_quat()  # (T, 4) xyzw

    imu_lfa = apply_lfa(imu, quats)

    # The LAST sample should be unchanged (R_{T-1}_last = I)
    last_diff = np.abs(imu_lfa[:, -1] - imu[:, -1]).max()
    print(f"|LFA(imu)[:, -1] - imu[:, -1]|_max = {last_diff:.2e}  (should be ~0)")
    assert last_diff < 1e-10, "LFA last-frame anchor broken!"
    print("LFA last-frame anchor verified.")
