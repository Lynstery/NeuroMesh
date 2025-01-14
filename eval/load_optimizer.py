import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import threading
import multiprocessing
from mesh.eval.load_predictor import TransformerPredictor, num_epochs

class CameraInfo:
    def __init__(self, cam_id, decode_latency, past_loads, nearby_loads):
        self.cam_id = cam_id
        self.decode_latency = decode_latency

        self.past_loads = past_loads
        self.nearby_loads = nearby_loads
        self.workload = past_loads[-1]
        self.future_loads = None

        self.balance_weight = 0

    def set_decode_latency(self, decode_latency):
        self.decode_latency = decode_latency
     
    def update_load(self, new_load):
        self.past_loads.append(new_load)
        self.past_loads = self.past_loads[1:]
        self.workload = new_load
    
class Monitor:
    def __init__(self, decode_latencies, bandwidth_matrix):
        self.num_cameras = len(decode_latencies)
        load_len = 8 
        self.cameras_info = [CameraInfo(cam_id=i, decode_latency=decode_latencies[i],
                                        past_loads=[7 * random.random() for _ in range(load_len)],
                                        nearby_loads=[8 * random.random() for _ in range(load_len)])
                            for i in range(self.num_cameras)]
        self.decode_latencies = decode_latencies
        self.bandwidth_matrix = bandwidth_matrix

        self.update_queue = multiprocessing.Queue()
        self.lock = threading.Lock()
        self.predictor = None
    
    def load_predicter(self):
        # future loads predictor
        self.predictor = TransformerPredictor(
            input_dim=2, model_dim=64, num_heads=4, num_layers=2, output_dim=1)
        #self.predictor.load_model(num_epochs)
        
    def update_bandwidth_matrix(self, src_id, bandwidths):
        self.update_queue.put((1, (src_id, bandwidths)))

    def update_workload(self, cam_id, workload):
        self.update_queue.put((2, (cam_id, workload)))
    
    def run_update_loop(self):
        while True:
            type, elems = self.update_queue.get()
            if type == 1:
                src_id, bandwidths = elems
                for dst_id in range(len(bandwidths)):
                    self.bandwidth_matrix[src_id][dst_id] = bandwidths[dst_id]
            else:
                cam_id, workload = elems
                self.cameras_info[cam_id].update_load(workload)
        
    def get_bandwidth_edge_cam(self, cam_id): 
        return self.bandwidth_matrix[cam_id][cam_id]
    
    def get_decode_latencies(self):
        return self.decode_latencies.copy()
         
    def get_bandwidth_matrix(self):
        return self.bandwidth_matrix.copy()
        
    def get_cameras_workloads(self):
        return np.array([cam.workload for cam in self.cameras_info])

    def get_cameras_predicted_workloads(self):
        workloads = np.array([cam.workload for cam in self.cameras_info])
        '''
            past_loads_list = [] 
            for idx, camera in enumerate(self.cameras_info):
                device_past_loads = torch.tensor(camera.past_loads, dtype=torch.float32)
                nearby_past_loads = torch.tensor(camera.nearby_loads, dtype=torch.float32)
                past_loads = torch.stack((device_past_loads, nearby_past_loads), dim=-1).unsqueeze(0)
                past_loads_list.append(past_loads)   

        for idx, camera in enumerate(self.cameras_info):
            past_loads = past_loads_list[idx]
            camera_future_loads = self.predictor(past_loads).tolist()[0]
            workloads[idx] += camera_future_loads[0] # add future workload
        ''' 
        return workloads
        
class DistreamLoadBalancer:
    def __init__(self):
        self.imbalance_index = 0.3

    def compute_imbalance_index(self, workloads, capacities):
        max_val = 0.0
        sum_val = 0.0
        n = len(workloads)
        for i in range(n):
            tmp = workloads[i] / capacities[i]
            if tmp > max_val:
                max_val = tmp
            sum_val += tmp
        if sum_val > 0:
            a = max_val / (sum_val / n)
            return a - 1
        else:
            return 0.0

    def load_balance(self, workloads, capacities):
        num_cameras = len(workloads)
        migrations = np.zeros((num_cameras, num_cameras), dtype=np.float32)

        v = self.compute_imbalance_index(workloads, capacities)
        while v > self.imbalance_index:
            max_val = -1.0
            max_idx = -1
            min_val = 10000.0
            min_idx = -1
            n = len(workloads)
            for i in range(n):
                tmp = workloads[i] / capacities[i]
                if tmp > max_val:
                    max_val = tmp
                    max_idx = i
                if tmp < min_val:
                    min_val = tmp
                    min_idx = i
            if (workloads[max_idx] - workloads[min_idx]) < 4:
                break
            num_to_migrate = 2
            migrations[max_idx, min_idx] += num_to_migrate
            workloads[max_idx] -= num_to_migrate
            workloads[min_idx] += num_to_migrate
            v = self.compute_imbalance_index(workloads, capacities)

        return migrations

class MeshLoadBalancer:
    def __init__(self):
        pass

    def calculate_balance_weights(self, decode_latencies):
        avg_latency = np.mean(decode_latencies)
        balance_weights = decode_latencies / avg_latency
        return balance_weights

    def calculate_imbalance_index(self, workloads, balance_weights):
        weighted_workloads = workloads * balance_weights 
        max_weighted_workload = np.max(weighted_workloads)
        sum_weighted_workload = np.sum(weighted_workloads)
        return (len(workloads) * max_weighted_workload / sum_weighted_workload) - 1

    def get_max_bandwidth(self, matrix):
        row_maxes = [max(row) for row in matrix]
        return max(row_maxes)

    def balance_load(self, workloads, decode_latencies, bandwidth_matrix):
        '''
            calculate the migration amount for each camera base on current states
        '''
        num_cameras = len(workloads)
        migrations = np.zeros((num_cameras, num_cameras), dtype=np.float32)

        balance_weights = self.calculate_balance_weights(decode_latencies)
        previous_imbalance_index = self.calculate_imbalance_index(workloads, balance_weights)
        no_improvement_count = 0

        while True:
            weighted_workloads = workloads * balance_weights
            id_max = np.argmax(weighted_workloads)
            id_min = np.argmin(weighted_workloads)

            bandwidth = bandwidth_matrix[id_max][id_min]
            balance_amount = (weighted_workloads[id_max] - weighted_workloads[id_min]) / (balance_weights[id_min] + balance_weights[id_max])
            max_bandwidth = self.get_max_bandwidth(bandwidth_matrix)
            migration_amount = balance_amount * (bandwidth / max_bandwidth) # + backlog_loads[id_max]
            #backlog_loads[id_max] = 0

            migrations[id_max][id_min] += migration_amount
            workloads[id_max] -= migration_amount
            workloads[id_min] += migration_amount

            #print(f"Workload {migration_amount} from Camera {id_max} to Camera {id_min}")

            self.current_imbalance_index = self.calculate_imbalance_index(workloads, balance_weights)
            
            if self.current_imbalance_index >= previous_imbalance_index:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if no_improvement_count >= 2:
                break
            
            #latest_bandwidth = random.uniform(5, 15)
            #latest_max_bandwidth = random.uniform(latest_bandwidth, 15)

            #if latest_bandwidth / latest_max_bandwidth > bandwidth / max_bandwidth:
            #    backlog_loads[id_max] = balance_amount * (latest_bandwidth / latest_max_bandwidth - bandwidth / max_bandwidth)

            previous_imbalance_index = self.current_imbalance_index

        print("balanced workloads:", workloads)
        return migrations 
  
class InferPropagateController:
    def __init__(self, num_cameras, similar_pair_num=3546, similar_mean=0.36, similar_std=0.14):
        self.num_cameras = num_cameras
        self.similar_pair_num = similar_pair_num
        self.similar_mean = similar_mean
        self.similar_m2 = 0.0
        self.similar_std = similar_std
        self.threshold = similar_mean + similar_std

        self.manager = multiprocessing.Manager()
        self.cameras_result = self.manager.list([None for _ in range(num_cameras)])
        self.cameras_patches = self.manager.list([None for _ in range(num_cameras)])
        self.anchors = self.manager.list([None for _ in range(num_cameras)])
        
    def cosine_similarity(self, tensor1, tensor2):
        tensor1_flat = tensor1.view(-1)
        tensor2_flat = tensor2.view(-1)
        if tensor1_flat.shape[0] != tensor2_flat.shape[0]:
            max_length = max(tensor1_flat.shape[0], tensor2_flat.shape[0])
            tensor1_flat = F.interpolate(tensor1_flat.unsqueeze(0).unsqueeze(
                0), size=max_length, mode='linear', align_corners=False).squeeze()
            tensor2_flat = F.interpolate(tensor2_flat.unsqueeze(0).unsqueeze(
                0), size=max_length, mode='linear', align_corners=False).squeeze()
        cosine_sim = F.cosine_similarity(
            tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0))
        return cosine_sim.item()

    def check_overlap(self, current_camid, current_patch):
        similarities = [(i, self.cosine_similarity(current_patch, self.cameras_patches[i]))
                        for i in range(self.num_cameras) if i != current_camid and self.cameras_patches[i] is not None]
        if not similarities:
            return -2, -2
        anchor_camid, anchor_similar = max(similarities, key=lambda x: x[1], default=(0, 0))
        return anchor_camid, anchor_similar

    def update_threshold(self, similarity):
        # Welford's online algorithm
        self.similar_pair_num += 1
        old_mean = self.similar_mean
        self.similar_mean += (similarity - old_mean) / self.similar_pair_num
        self.similar_m2 += (similarity - old_mean) * (similarity - self.similar_mean)
        if self.similar_pair_num > 1:
            self.similar_std = (
                self.similar_m2 / (self.similar_pair_num - 1)) ** 0.5
        else:
            self.similar_std = 0.0
        self.threshold = self.similar_mean + self.similar_std

    def update_result_propagation(self, current_camid):
        current_patch = self.cameras_patches[current_camid]
        if current_patch is not None:
            anchor_camid, anchor_similar = self.check_overlap(current_camid, current_patch)
            if anchor_camid != -2:
                if anchor_similar > self.threshold:
                    self.anchors[current_camid] = anchor_camid
                    self.update_threshold(anchor_similar)
                else:
                    self.anchors[current_camid] = None


if __name__ == "__main__":
    num_cameras = 5
    
    decode_latencies = np.array([random.uniform(0.01, 0.01) for _ in range(num_cameras)])
    
    bandwidth_matrix = np.zeros((num_cameras, num_cameras), dtype=np.float32)
    for i in range(num_cameras):
        for j in range(i + 1, num_cameras):
            value = random.uniform(10, 10)
            bandwidth_matrix[i][j] = value
            bandwidth_matrix[j][i] = value

    workloads = np.array([random.uniform(0, 10) for _ in range(num_cameras)])
    print("workload:", workloads) 
    optimizer = MeshLoadBalancer()
    
    migrations = optimizer.balance_load(workloads, decode_latencies, bandwidth_matrix)
    
    for i in range(num_cameras):
        for j in range(num_cameras):
            print(f"{migrations[i][j]:.2f} ", end="")
        print("")
