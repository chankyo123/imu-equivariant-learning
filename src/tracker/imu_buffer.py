import numpy as np
from scipy.interpolate import interp1d


class ImuBuffer:
    """ This is a buffer for interpolated IMU data."""

    def __init__(self):
        self.net_t_us = np.array([], dtype=int)
        self.net_acc = np.array([])
        self.net_gyr = np.array([])

    def add_data_interpolated(
        self, last_t_us, t_us, last_gyr, gyr, last_acc, acc, requested_interpolated_tus
    ):
        assert isinstance(last_t_us, int)
        assert isinstance(t_us, int)
        if last_t_us < 0:
            acc_interp = acc.T
            gyr_interp = gyr.T
        else:
            try:
                acc_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_acc.T, acc.T]),
                    axis=0,
                )(requested_interpolated_tus)
                gyr_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_gyr.T, gyr.T]),
                    axis=0,
                )(requested_interpolated_tus)
            except ValueError as e:
                print(
                    f"Trying to do interpolation at {requested_interpolated_tus} between {last_t_us} and {t_us}"
                )
                raise e
        self._add_data(requested_interpolated_tus, acc_interp, gyr_interp)

    def _add_data(self, t_us: int, acc, gyr):
        assert isinstance(t_us, int)
        if len(self.net_t_us) > 0:
            assert (
                t_us > self.net_t_us[-1]
            ), f"trying to insert a data at time {t_us} which is before {self.net_t_us[-1]}"

        self.net_t_us = np.append(self.net_t_us, t_us)
        self.net_acc = np.append(self.net_acc, acc).reshape(-1, 3)
        self.net_gyr = np.append(self.net_gyr, gyr).reshape(-1, 3)

    # get network data by input size, extract from the latest
    def get_last_k_data(self, size):
        net_acc = self.net_acc[-size:, :]
        net_gyr = self.net_gyr[-size:, :]
        net_t_us = self.net_t_us[-size:]
        return net_acc, net_gyr, net_t_us

    # get network data from beginning and end timestamps
    def get_data_from_to(self, t_begin_us: int, t_us_end: int, update_freq: int):
        """ This returns all the data from ts_begin to ts_end """
        assert isinstance(t_begin_us, int)
        assert isinstance(t_us_end, int)
        update_freq = int(update_freq)
        
        begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        end_idx = np.where(self.net_t_us == t_us_end)[0][0]
        if update_freq == 1000:
            indices = np.arange(begin_idx + 5, end_idx + 10, 5)
        else:
            indices = np.arange(begin_idx, end_idx + 1, 1)
            
        #     print(begin_idx, end_idx)  #0 995 / 1 200
        #     begin_idx = int((begin_idx+5)/5)
        #     end_idx = int((end_idx+5)/5)
        
        # abs_diff = np.abs(self.net_t_us - t_begin_us)
        # begin_idx = np.argmin(abs_diff)
        # end_idx = begin_idx + 199
        # abs_diff_end = np.abs(self.net_t_us - t_us_end)
        # end_idx = np.argmin(abs_diff_end)
        # begin_idx = end_idx - 199
        # print(t_begin_us * 1e-6, begin_idx, self.net_t_us.shape[0])
        # assert False
        
        # print("inside buffer : ", self.net_acc.shape, self.net_gyr.shape, self.net_t_us.shape)
        # net_acc = self.net_acc[begin_idx : end_idx + 1, :]
        # net_gyr = self.net_gyr[begin_idx : end_idx + 1, :]
        # net_t_us = self.net_t_us[begin_idx : end_idx + 1]
        net_acc = self.net_acc[indices]
        net_gyr = self.net_gyr[indices]
        net_t_us = self.net_t_us[indices]
        
        # print(t_begin_us, net_t_us[:3])
        
        return net_acc, net_gyr, net_t_us

    def throw_data_before(self, t_begin_us: int):
        """ throw away data with timestamp before ts_begin
        """
        assert isinstance(t_begin_us, int)
        # begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        abs_diff = np.abs(self.net_t_us - t_begin_us)
        begin_idx = np.argmin(abs_diff)
        
        self.net_acc = self.net_acc[begin_idx:, :]
        self.net_gyr = self.net_gyr[begin_idx:, :]
        self.net_t_us = self.net_t_us[begin_idx:]

    def total_net_data(self):
        return self.net_t_us.shape[0]

    def debugstring(self, query_us):
        print(f"min:{self.net_t_us[0]}")
        print(f"max:{self.net_t_us[-1]}")
        print(f"que:{query_us}")
        print(f"all:{self.net_t_us}")
