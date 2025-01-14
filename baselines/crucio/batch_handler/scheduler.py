import os
import time

import numpy as np
import torch
from scipy.optimize import Bounds, curve_fit, minimize

from crucio.autoencoder.dataset import VIDEO_DIR, load_video_to_tensor
from crucio.autoencoder.util import save_compressed_data
from crucio.preheat import adjust_batch_size, crucio_delay, encoder3d


def func_d(b, delta1, delta2, delta3):
    return delta1*np.log(delta2*b)+delta3


def func_bopt(F, m, n):
    return m*F+n


class BatchScheduler():
    def __init__(self):
        # Initial batch size
        self.b_0 = 2
        # Time slot (s)
        self.T = 1.5
        # Benchmark bandwidth (Mbps)
        self.B_bench = 1
        # Small batch size
        self.b_small = self.b_0
        # Large batch size
        self.b_large = self.b_0
        # Optimal batch size
        self.b_opt = self.b_0
        # Parameter α>0
        self.alpha = 0
        # Parameter β>1
        self.beta = 1.5
        # Parameter 0<γ<1
        self.gamma = 0.5
        # Parameter m
        self.m = 0
        # Parameter n
        self.n = 0
        # Parameter δ1
        self.delta1 = 3
        # Parameter δ2>0
        self.delta2 = 2
        # Parameter δ3
        self.delta3 = 3

    def compute_d(self, b):
        return self.delta1*np.log(self.delta2*b)+self.delta3

    def fit_d(self, b_array, d_array):
        bounds = Bounds([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
        delta, _ = curve_fit(func_d, b_array, d_array, bounds=bounds)
        self.delta1 = delta[0]
        self.delta2 = delta[1]
        self.delta3 = delta[2]

    def fit_b(self, F_array, b_array):
        bounds = Bounds([0, self.b_0/2], [np.inf, np.inf])
        mn = curve_fit(func_bopt, F_array, b_array, bounds=bounds)[0]
        self.m = mn[0]
        self.n = mn[1]
        self.beta = 2*self.n/self.b_0
        assert self.beta >= 1
        def obj_func(x): return x[0]-self.T*x[1]
        cons = ({'type': 'eq', 'fun': lambda x: x[0] + self.T*x[1] - 2*self.m})
        bounds = Bounds([0, 0], [np.inf, 1])
        x0 = [1, 0.5]
        res = minimize(obj_func, x0, method='SLSQP',
                       bounds=bounds, constraints=cons)
        self.alpha = res.x[0]
        self.gamma = res.x[1]

    def compute_bmin(self, F):
        return self.alpha*F+self.beta*self.b_0

    def compute_bmax(self, F):
        return self.gamma*F*self.T

    def solve_batch_size(self, F, B, b_array, d_array):
        self.fit_d(b_array, d_array)
        self.b_small = self.compute_bmin(F)
        self.b_large = self.compute_bmax(F)
        self.b_opt = (self.b_small+self.b_large)/2
        b = self.b_opt
        if B > self.B_bench:
            B_incre = B/self.B_bench-1
            while b <= self.b_large:
                b += 1
                batch_decre = F*self.T/self.b_opt-F*self.T/b
                if batch_decre >= B_incre:
                    break
        elif B < self.B_bench:
            B_decre = (self.B_bench-B)/self.B_bench
            while b >= self.b_small:
                b -= 1
                d_decre = (self.compute_d(self.b_opt) -
                           self.compute_d(b))/self.compute_d(self.b_opt)
                if d_decre >= B_decre:
                    break
        return round(b)


def offline_profiling(video_name, time_slot, bandwidth, point=15):
    F_array = [16+_ for _ in range(point)]
    b_array = []
    for _ in range(point):
        frame_rate = F_array[_]
        print(f"Analyzing frame_rate={frame_rate}")
        length = 12
        batch_size = frame_rate-length
        optimal = batch_size
        min_delay = 999
        for _ in range(length):
            # print(f"Analyzing batch_size={batch_size}")
            frame_num = int(time_slot*frame_rate)
            if batch_size != adjust_batch_size(frame_num, batch_size):
                batch_size += 1
                continue
            filter_ratio, encoding_time, data_size, decoding_time, infer_time = crucio_delay(
                video_name, frame_num, batch_size)
            delay = encoding_time+data_size / \
                (bandwidth*1024/8)+decoding_time+infer_time
            if delay < min_delay:
                min_delay = delay
                optimal = batch_size
            batch_size += 1
        print(f"Optimal batch size={optimal}")
        b_array.append(optimal)
    return F_array, b_array


profiled_F_array = [16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
profiled_b_array = [5, 5, 7, 7, 8, 13, 13, 13, 13, 13, 16, 16, 16, 19, 19]


Scheduler = BatchScheduler()
Scheduler.fit_b(profiled_F_array, profiled_b_array)


def online_profiling(video_name, point=2):
    video_path = VIDEO_DIR+'/'+video_name
    b_array = [8+_*2 for _ in range(point)]
    d_array = []
    for _ in range(point):
        video_tensor = load_video_to_tensor(
            video_path, length=b_array[_]).unsqueeze(0)
        with torch.no_grad():
            compressed_data = encoder3d(video_tensor)
        data_path = save_compressed_data(video_path, compressed_data)
        data_size = os.path.getsize(data_path)/1024
        d_array.append(data_size)
        os.remove(data_path)
    return b_array, d_array


if __name__ == '__main__':
    F_array, b_array = offline_profiling(
        'car_0008_015', Scheduler.T, Scheduler.B_bench)
    Scheduler.fit_b(F_array, b_array)
    start = time.time()
    # b_array, d_array = online_profiling('aeroplane_0004_026')
    # b_array, d_array = online_profiling('bird_0015_016')
    # b_array, d_array = online_profiling('cat_0021_003')
    # b_array, d_array = online_profiling('boat_0016_023')
    # b_array, d_array = online_profiling('train_0030_006')
    # b_array, d_array = online_profiling('car_0009_015')
    # b_array, d_array = online_profiling('cow_0005_024')
    # b_array, d_array = online_profiling('motorbike_0011_017')
    # b_array, d_array = online_profiling('horse_0006_016')
    b_array, d_array = online_profiling('dog_0033_006')
    batch_size = Scheduler.solve_batch_size(
        30, Scheduler.B_bench*2, b_array, d_array)
    print(f"Optimal batch size is {batch_size}")
    end = time.time()
    overhead = end-start
    print(f"Scheduling overhead is {overhead:.4f} s")
