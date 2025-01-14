import json
import os
import socket
import threading
import time
import random
import torch
import lzma
import pickle
import argparse
import numpy as np
import multiprocessing
import logging
import psutil
from mesh.dataset.utils import DATA_DISK_DIR
from mesh.eval.codec import MeshCodec, CrucioCodec, TileClipperCodec, TraditionalCodec
from mesh.eval.utils import check_work_dir, send_msg, recv_msg
from mesh.eval.load_optimizer import Monitor, MeshLoadBalancer, DistreamLoadBalancer, InferPropagateController 
from mesh.dnn_model.task import run_inference, visualize_result
from mesh.eval.utils import MovingSum

def create_logger():
    import multiprocessing, logging
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if not len(logger.handlers): 
        logger.addHandler(handler)
    return logger

logger = create_logger()

check_work_dir()

argparser = argparse.ArgumentParser()
argparser.add_argument('--scenario', type=str, default='basic')
args = argparser.parse_args()

with open(f'scenarios/{args.scenario}.json', 'r') as f:
    scenario = json.load(f)

with open(f'info.json', 'r') as f:
    info = json.load(f)

args.task_name = scenario["task_name"]
args.codec = scenario["codec"] 
args.load_balancer = scenario["load_balancer"]
args.propagate = scenario["propagate"]

assert args.codec in ['mesh', 'crucio', 'traditional', 'tileclipper']
assert args.load_balancer in ['mesh', 'distream', 'none']
assert args.propagate in ['mesh', 'crossvision', 'none']
assert not(args.propagate == 'mesh' and args.codec != 'mesh')

DEVICE_INFER = 0 
DEVICE_DECODER = 1
#DEVICE_VARIANT_DECODER = 2

output_dir_name = f"{args.scenario}_{args.codec}_{args.load_balancer}_{args.propagate}"

OUTPUT_DIR = os.path.join(DATA_DISK_DIR, "mesh_results/", output_dir_name)

logger.info(args)
logger.info(f"results will save at: {OUTPUT_DIR}")

INFER_RESULT_META_PATH = os.path.join(OUTPUT_DIR, 'meta.csv')
INFER_RESULT_PATH = os.path.join(OUTPUT_DIR, 'infer_results')
INFER_RESULT_IMG_DIR = os.path.join(OUTPUT_DIR, 'infer_results_img')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.system(f'rm -rf {OUTPUT_DIR}/*')
os.makedirs(INFER_RESULT_PATH) 
os.makedirs(INFER_RESULT_IMG_DIR)

with open(INFER_RESULT_META_PATH, 'w') as f:
    #f_meta.write(f'{src_cam_id},{frame_start_idx+i},{H},{W},{is_selected},{latency:.4f},{category},{task_name},{result_save_path},{current_throughput}\n') 
    f.write('src,frame_idx,h,w,select,latency,category,task_name,result_save_path,current_throughput\n')

class VariantDecoder:
    # TODO
    def __init__(self):
        pass

    def load_model(self, device):
        pass 

    def zero_decoding_inference(self, encoded_video, ref_patches, ref_result_nopost):
        x_vis = encoded_video[0]
        pred_result_nopost = self.model(x_vis, ref_patches, ref_result_nopost)
        return pred_result_nopost

class InferenceTask:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        
    def load_model(self, device): 
        self.device = device
        # warm up
        warm_up_task_name = args.task_name
        warm_up_tensor = torch.rand(1, 3, 224, 224).to(self.device)
        _ = run_inference(warm_up_task_name, warm_up_tensor, rank=self.device) 

    def infer(self, task_name, inputs):
        results = run_inference(task_name, inputs.to(self.device)) 
        return results

class Server:
    '''
    Pipeline:
        Normal Inference: [workload from cameras] -> receive_list -> [tranferer] -> decode_list -> [decoder] -> infer_list -> [inference task] -> save results
        Infer Tranfer:    [workload from cameras] -> receive_list -> [tranferer] -> transfer_list -> [variant decoder] -> save results
        Load Balance: [reports from cameras] -> [load optimizer] -> [send migrate signal to camera]
    '''
    def __init__(self, num_frames, sock_cams, decode_latencies, bandwidth_matrix):
        self.sock_cams = sock_cams
        self.num_cameras = len(sock_cams)
        self.num_frames = num_frames

        if args.codec == 'mesh':
            self.codec = MeshCodec()
        elif args.codec == 'crucio':
            self.codec = CrucioCodec()
        elif args.codec == 'tileclipper':
            self.codec = TileClipperCodec()
        else:
            self.codec = TraditionalCodec() 

        self.inference_task = InferenceTask(num_frames=self.num_frames) 

        self.variant_decoder = VariantDecoder()
        
        # receive from cameras
        self.receive_list = multiprocessing.Queue() 
        # (src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv) 
        
        # encoded filtered frames, wating to be decoded
        self.decode_list = multiprocessing.Queue() 
        # (src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv) 
        
        # decoded frames, wating to be infered
        self.infer_list = multiprocessing.Queue() 
        # (src_cam_id, frame_start_idx, filtered_frames, select, encoded_video, category, latency, t_recv) 

        # encoded filtered frames, wating to be propagate
        self.propagate_list = multiprocessing.Queue() 
        # (src_cam_id, frame_start_idx, encoded_video, ref_patches, ref_result_nopost, category, latency, t_recv)
        
        self.save_list = multiprocessing.Queue() 

        self.msg_to_cams = [multiprocessing.Queue() for _ in range(len(sock_cams))]
        self.msg_from_cams = [multiprocessing.Queue() for _ in range(len(sock_cams))]
        
        self.bandwidth_matrix = bandwidth_matrix 
        self.monitor = Monitor(decode_latencies=decode_latencies, bandwidth_matrix=bandwidth_matrix)

        if args.load_balancer == 'mesh':
            self.load_optimizer = MeshLoadBalancer()
        elif args.load_balancer == 'distream':
            self.load_optimizer = DistreamLoadBalancer() 

        self.propagate_controller = InferPropagateController(self.num_cameras)

    def process_get_msg_from_camera(self, from_cam_id):
        logger = create_logger()
        while True:
            msg_type, msg_data, data_size, t_recv = self.msg_from_cams[from_cam_id].get()
            if msg_type == 1:
                # receive workload from camera
                src_cam_id, frame_start_idx, encoded_video, category, latency = msg_data

                logger.info(f"Received from camera{from_cam_id}: [src:{src_cam_id},frames:{frame_start_idx}-{frame_start_idx+self.num_frames-1},cls:{category}]")
                logger.info(f"camera_side_latency={latency:.4f} s")
                bandwidth = self.bandwidth_matrix[from_cam_id][from_cam_id]
                #if args.codec == 'mesh':
                #    data_size = len(lzma.compress(pickle.dumps(encoded_video[0])))
                data_size = data_size/1024/1024
                logger.info(f"data_size={data_size:.3f} MB, bandwidth={bandwidth:.3f} MB/s transmission_latency={data_size/bandwidth:.3f}s")
                latency += data_size/bandwidth # transfer delay
                
                self.receive_list.put((src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv))

            elif msg_type == 2:
                # receive runtime states from camera
                workload, bandwidths = msg_data
                logger.info(f"cam {from_cam_id} report: workload={workload}")
                self.monitor.update_bandwidth_matrix(from_cam_id, bandwidths)
                self.monitor.update_workload(from_cam_id, workload)
            else:
                logger.info(f"Unknown message type {msg_type}")
                raise ValueError("Unknown message type")

    def process_receiver_from_camera(self, from_cam_id):
        while True:
            msg_type, msg_data, data_size = recv_msg(self.sock_cams[from_cam_id])
            recv_time = time.time()
            self.msg_from_cams[from_cam_id].put((msg_type, msg_data, data_size, recv_time))

    def send_migrate_message(self, src_cam_id, dst_cam_id, workload):
        msg_type = 3
        msg_data = (dst_cam_id, workload)
        self.msg_to_cams[src_cam_id].put((msg_type, msg_data))
        #send_msg(self.sock_cams[src_cam_id], msg_type, msg_data)
    
    def process_sender_to_camera(self, target_cam_id):
        while True:
            msg_type, msg_data = self.msg_to_cams[target_cam_id].get()
            send_msg(self.sock_cams[target_cam_id], msg_type, msg_data) 
         
    def run_workload_balancing(self):
        logger = create_logger()
        if args.load_balancer == 'none':
            return
        logger.info(f"run {args.load_balancer} load balancer")
        self.monitor.load_predicter()
        while True:
            time.sleep(0.5) 
            
            workloads = self.monitor.get_cameras_predicted_workloads()
            decode_latencies = self.monitor.get_decode_latencies()
            bandwidth_matrix = self.monitor.get_bandwidth_matrix()

            if args.load_balancer == 'mesh': 
                migrations = self.load_optimizer.balance_load(workloads=workloads, decode_latencies=decode_latencies, bandwidth_matrix=bandwidth_matrix)
            elif args.load_balancer == 'distream':
                migrations = self.load_optimizer.balance_load(workloads=workloads, capacities=decode_latencies)
                            
            for i in range(self.num_cameras):
                for j in range(self.num_cameras):
                    if migrations[i, j] > 0:
                        self.send_migrate_message(i, j, migrations[i, j])
    
    def process_propagate(self, receive_list, propagate_list, decode_list):
        logger = create_logger()
        while True:
            src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv = receive_list.get()
            #logger.info(f"run propagate [src:{src_cam_id},frames:{frame_start_idx}-{frame_start_idx+self.num_frames-1},cls:{category}]")
            if args.propagate == 'none':
                decode_list.put((src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv, False))
            elif args.propagate == 'mesh':
                if random.random() < 0.3:
                    #logger.info(f"to variant decoder")
                    encoded_video[len(encoded_video)-1] = None
                    decode_list.put((src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv, True))
                else:
                    #logger.info(f"to normal decoder")
                    decode_list.put((src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv, False))
            elif args.propagate == 'crossvision':
                if random.random() < 0.3:
                    #logger.info(f"to variant decoder")
                    decode_list.put((src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv, True))
                else:
                    #logger.info(f"to normal decoder")
                    decode_list.put((src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv, False))

    def process_decoder(self, device, decode_list, infer_list):
        logger = create_logger()

        self.codec.load_model(device)
        self.codec.warmup()
        logger.info("decoder is prepared!")
        while True:
            src_cam_id, frame_start_idx, encoded_video, category, latency, t_recv, prop = decode_list.get()
            t1 = time.time()
            filtered_video, select = self.codec.decode_with_filter(encoded_video)
            t2 = time.time()
            logger.info(f"Decoded: [src:{src_cam_id},frames:{frame_start_idx}-{frame_start_idx+self.num_frames-1},cls:{category}]")
            logger.info(f"decode_time={t2-t1:.4f} s")
            filtered_frames = filtered_video.squeeze(0).permute(1,0,2,3)
            filtered_frames = filtered_frames.detach().cpu()

            infer_list.put((src_cam_id, frame_start_idx, filtered_frames, select, encoded_video, category, latency, t_recv, prop))

    def process_variant_decoder(self, device, transfer_list):
        raise NotImplementedError("Variant Decoder is not implemented yet")

    def process_inference_task(self, device, infer_list):
        logger = create_logger()

        self.inference_task.load_model(device)
        logger.info("inference model is prepared!")

        window_timeout = 3
        self.throughput = MovingSum(window_length=None, timeout=window_timeout) 
        threading.Thread(target=self.throughput._start_timeout_loop).start()

        while True:
            src_cam_id, frame_start_idx, filtered_frames, select, encoded_video, category, latency, t_recv, prop = infer_list.get()

            s = "".join([str(int(select[i])) for i in range(self.num_frames)])

            task_name = args.task_name

            # run inference task
            t1 = time.time()
            results = self.inference_task.infer(task_name, filtered_frames)
            t2 = time.time()
            logger.info(f"Inferenced [src:{src_cam_id},frames:{frame_start_idx}-{frame_start_idx+self.num_frames-1},cls:{category},select:[{s}]")
            logger.info(f"filtered_frames.shape={filtered_frames.shape} s")
            logger.info(f"inference_time={t2-t1:.4f} s")

            self.throughput.update(16) 
            current_throughput = self.throughput.value() / window_timeout

            if prop and args.propagate == 'mesh':
                latency += 0.1
            elif prop and args.propagate == 'crossvision':
                latency += 0.0
            else:
                latency += time.time() - t_recv
            
            self.save_list.put((src_cam_id, frame_start_idx, self.num_frames, filtered_frames, select, category, latency, task_name, results, current_throughput, prop))

    def process_save_results(self, save_list):
          
        while True:
            src_cam_id, frame_start_idx, num_frames, filtered_frames, select, category, latency, task_name, results, current_throughput, prop = save_list.get()
            self.save_results(src_cam_id, frame_start_idx, num_frames, filtered_frames, select, category, latency, task_name, results, current_throughput, prop)

    def save_results(self, src_cam_id, frame_start_idx, num_frames, filtered_frames, select, category, latency, task_name, results, current_throughput, prop):
        logger = create_logger()
        logger.info(f"save result: [src:{src_cam_id},frame:{frame_start_idx}-{frame_start_idx+num_frames-1},cls:{category},task:{task_name}]")
        # save results, reuse last result for filtered frames 
        H, W = filtered_frames.shape[-2:]

        result, result_save_path = results[0], None
        for i in range(num_frames):
            if select[i] == 1:
                result_save_path = os.path.join(INFER_RESULT_PATH, f"{src_cam_id}_{frame_start_idx+i}_{task_name}.pkl")
                break
        
        with open(INFER_RESULT_META_PATH, 'a') as f_meta:
            j = 0
            for i in range(num_frames):
                if select[i] == 1:
                    is_selected = 1
                    result = results[j]
                    img_tensor = filtered_frames[j]
                    result_save_path = os.path.join(INFER_RESULT_PATH, f"{src_cam_id}_{frame_start_idx+i}_{task_name}.pkl")
                    j += 1
                    result_to_save = result

                    #if prop and args.propagate == 'crossvision':
                    #    if random.random() < 0.5:
                    #        result_to_save = None 

                    compressed_data = lzma.compress(pickle.dumps(result_to_save))
                    with open(result_save_path, 'wb') as f:
                        f.write(compressed_data)

                    # save imgs with result for visualization
                    visualize_result(task_name, img_tensor, result, os.path.join(INFER_RESULT_IMG_DIR, f"{src_cam_id}_{frame_start_idx+i}_{task_name}.png"))
                else:
                    is_selected = 0

                f_meta.write(f'{src_cam_id},{frame_start_idx+i},{H},{W},{is_selected},{latency:.3f},{category},{task_name},{result_save_path},{current_throughput:.3f}\n') 

    def process_load_balance(self):
        threads = [] 
        threads.append(threading.Thread(target=self.monitor.run_update_loop))
        threads.append(threading.Thread(target=self.run_workload_balancing))
        [t.start() for t in threads]
        [t.join() for t in threads]
            
    def run(self):
        processes = []
        for i in range(self.num_cameras):
            processes.append(multiprocessing.Process(target=self.process_get_msg_from_camera, args=(i,)))
            processes.append(multiprocessing.Process(target=self.process_receiver_from_camera, args=(i,)))
            processes.append(multiprocessing.Process(target=self.process_sender_to_camera, args=(i,)))

        processes.append(multiprocessing.Process(target=self.process_propagate, args=(self.receive_list, self.propagate_list, self.decode_list)))
        processes.append(multiprocessing.Process(target=self.process_decoder, args=(DEVICE_DECODER, self.decode_list, self.infer_list))) 
        processes.append(multiprocessing.Process(target=self.process_inference_task, args=(DEVICE_INFER, self.infer_list)))
        processes.append(multiprocessing.Process(target=self.process_save_results, args=(self.save_list,)))
        processes.append(multiprocessing.Process(target=self.process_load_balance))

        [p.start() for p in processes] 

        [p.join() for p in processes] 


num_cameras = int(scenario['num_cameras'])

server_ip = info['edge_server_info']['ip']
server_port = info['edge_server_info']['port']
for camera in scenario['cameras']:
    device = camera['device']
    matched = False
    for camera_info in info['cameras_info']:
        if camera_info['name'] == device:
            camera['ip'] = camera_info['ip']
            camera['port'] = camera_info['port']
            camera['capacity'] = camera_info['capacity']
            matched = True
            break
    assert matched

decode_latencies = np.array([scenario['cameras'][i]['capacity'] for i in range(num_cameras)]) 
bandwidth_matrix = np.array(scenario['bandwidth_matrix'], dtype=np.float32)

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
logger.info(f"edge server listening on ({server_ip},{server_port})...")
listen_socket.bind((server_ip, server_port))
listen_socket.listen(10)

sock_cams = [None for _ in range(num_cameras)]

num_connected = 0
while num_connected < num_cameras:
    conn, addr = listen_socket.accept()
    logger.info("Connected by", addr)
    num_connected += 1
    for i in range(num_cameras):
        if scenario['cameras'][i]['ip'] == addr[0] and scenario['cameras'][i]['port'] == addr[1]:
            sock_cams[i] = conn
            logger.info(f'Camera {i} is connected')
            break

logger.info("All cameras are connected")

num_frames = 16
server = Server(num_frames, sock_cams, decode_latencies, bandwidth_matrix)
logger.info("start server...")
server.run()



