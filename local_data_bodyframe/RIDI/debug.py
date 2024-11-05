import os
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def bmv(R, v):
    return np.einsum('ijk,ik->ij', R, v)

def main():
    path_base = "local_data_bodyframe/RIDI"
    seqs = [f for f in os.listdir(path_base) if os.path.isdir(os.path.join(path_base, f))]
    print(seqs)
    for seq in seqs:
        path_seq = os.path.join(path_base, seq, "imu0_resampled.npy")
        data_seq: np.ndarray = np.load(path_seq)
        t = data_seq[:, 0] # us
        t = t / 1e6 # s
        dt = t[1] - t[0]
        print("dt: ", dt)
        w = data_seq[:, 1:4]
        a = data_seq[:, 4:7]
        q = data_seq[:, 7:11] # x, y, z, w
        p = data_seq[:, 11:14]
        v_body = data_seq[:, 14:17]

        quat = Rotation.from_quat(q, scalar_first=False)
        Rot = quat.as_matrix()
        v_test = bmv(Rot, v_body)

        print(Rot.shape)
        vel_world = (p[1:] - p[:-1]) / (t[1:] - t[:-1]).reshape(-1, 1)
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(t, v_test[:, i], label="v_test")
            ax[i].plot(t[1:], vel_world[:, i], label="v_world")
            ax[i].legend()
        fig.tight_layout()
        plt.show()
        


        break
        



if __name__ == "__main__":
    main()