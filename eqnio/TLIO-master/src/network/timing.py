import torch
import numpy as np


class CudaTimer:
    timers = {}
    def __init__(self, key):
        if key not in self.timers:
            self.timers[key] = []
        self.key = key

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        time_ms = self.start.elapsed_time(self.end)
        self.timers[self.key].append(time_ms)
        
    @classmethod
    def print_timings(cls):
        print("Timings")
        keys = sorted(list(cls.timers))
        for key in keys:
            print(f"\t {key} - {np.mean(cls.timers[key])}")