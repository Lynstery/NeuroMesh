'''
client 


send data structure:
    metadata: src_cam_id, frame_idx_first, frame_num, data_size, capture_time, send_time
    data: compressed data of video segment
'''

import json
import os
import socket
import struct
import time
import torch
import queue
import threading
import multiprocessing
import argparse
import lzma
import pickle
import subprocess
import numpy as np
import random
import signal
from torchvision import transforms
from mesh.eval.capture import ImageFolderVideoCapture
from mesh.eval.utils import check_work_dir, send_msg, recv_msg, connect_with_retry
from mesh.dataset.utils import DATA_DISK_DIR, timm_video_normalization, ReshapeVideo
from timm.models import create_model
from mesh.vitransformer.pretrain import __all__
from mesh.msc.engine_for_pretraining import generate_patchs_from_videos
from mesh.msc.engine_for_pretraining import (generate_patchs_from_videos, 
                                             reconstruct_videos_from_patchs_with_mean_std,
                                             video_frame_filtering)

from crucio.batch_handler.gru_filter import (get_filter,
                                             print_gru_info,
                                             scores_to_selects)
from crucio.autoencoder.network3d import get_networks3d
from mesh.dnn_model.util import get_categories
from mesh.dnn_model.video_classification import load_video_resnet
from mesh.eval.codec import MeshCodec, CrucioCodec, TileClipperCodec, TraditionalCodec
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

args.codec = scenario["codec"] 
args.load_balancer = scenario["load_balancer"]

assert args.codec in ['mesh', 'crucio', 'traditional', 'tileclipper']
assert args.load_balancer in ['mesh', 'distream', 'none']


num_cameras = int(scenario['num_cameras'])

for camera in scenario['cameras']:
    device = camera['device']
    for camera_info in info['cameras_info']:
        if camera_info['name'] == device:
            camera['ip'] = camera_info['ip']
            camera['port'] = camera_info['port']
            camera['capacity'] = camera_info['capacity']
            break

get_ip_result = subprocess.run(["bash", "../networking/get_ip.sh"], capture_output=True, text=True).stdout
get_ip_result = get_ip_result.strip()

for i, camera in enumerate(scenario['cameras']):
    if camera['ip'] == get_ip_result:
        args.cam_id = i

logger.info(args)

class VideoClassifier:
    def __init__(self):
        self.mu_load = threading.Lock()
        self.is_model_loaded = False
    
    def load_model(self, device):
        with self.mu_load:
            if not self.is_model_loaded:
                self.device = device
                self.is_model_loaded = True
                self.video_resnet_weights, self.video_resnet_running = load_video_resnet(rank=self.device)
    
    def classify(self, video_clip):
        logger = create_logger()
        video_clip = video_clip.to(self.device) 
        with torch.no_grad():
            results = self.video_resnet_running(video_clip)
        label, category, score = get_categories(self.video_resnet_weights, results, show=False)
        logger.info(f"classify: {label} {category} {score:.2f}")
        return label 
    
    def warmup(self):
        logger = create_logger()
        for _ in range(0, 5):
            video = torch.rand((1, 3, 16, 224, 224), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                results = self.video_resnet_running(video)
        logger.info("warmup finished!")

class Camera:
    '''
        Pipeline:
            To server: 
                [video capture] -> frame_list -> [group frames] -> clip_list -> [process] ->  send to edge or other cameras
            From other cameras:
                [from other cameras] -> decode_list -> [decode] -> clip_list
    ''' 
    
    def __init__(self, cam_id, num_frames, video_info, bandwidth_matrix, sock_edge, sock_cams):
        self.cam_id = cam_id
        self.num_frames = num_frames
        
        if args.codec == 'mesh':
            self.codec = MeshCodec()
        elif args.codec == 'crucio':
            self.codec = CrucioCodec()
        elif args.codec == 'tileclipper':
            self.codec = TileClipperCodec()
            self.codec.prepare(video_info['video_dir']) # prepare
        else:
            self.codec = TraditionalCodec() 

        self.classifier = VideoClassifier() 

        # frames captured from local camera with particular frame rate
        self.frame_list = multiprocessing.Queue() # (frame_idx, img_tensor)

        self.clip_list = multiprocessing.Queue() 
        # (src_cam_id, frame_start_idx, video, latency, t_recv)

        # encoded videos received from other cameras, waiting to be decoded
        self.decode_list = multiprocessing.Queue() 
        # (src_cam_id, frame_start_idx, encoded_video, latency, t_recv)

        self.video_dir = video_info['video_dir']
        self.frame_rate = float(video_info['frame_rate'])
        self.frame_total = video_info['frame_total']
        self.frame_size = (224, 224) # for tensor resizing
        self.video_capture = ImageFolderVideoCapture(self.video_dir, self.frame_rate, self.frame_total) 
        
        self.sock_edge = sock_edge
        self.sock_cams = sock_cams
        self.msg_to_edge = multiprocessing.Queue() 
        self.msg_to_cams = [multiprocessing.Queue() for _ in range(len(sock_cams))]
         
        self.bandwidths = [bandwidth_matrix[cam_id, i] for i in range(len(sock_cams))]
        self.migrate_list = multiprocessing.Queue() # migration signals from edge server
        
    def get_workload(self):
        return self.clip_list.qsize()*self.num_frames
    
    def update_bandwidth(self, target_cam_id, bandwidth):
        self.bandwidths[target_cam_id] = bandwidth
    
    def get_bandwidths(self):
        return list(self.bandwidths)

    def send_to_edge(self, msg_type, msg_data):
        self.msg_to_edge.put((msg_type, msg_data))

    def send_to_camera(self, target_cam_id, msg_type, msg_data):
        self.msg_to_cams[target_cam_id].put((msg_type, msg_data))
        
    def run_sender_to_edge(self):
        logger = create_logger()
        while True:
            msg_type, msg_data = self.msg_to_edge.get()
            send_msg(self.sock_edge, msg_type, msg_data)
            logger.info(f"msg sent")
    
    def run_sender_to_camera(self, target_cam_id):
        while True:
            msg_type, msg_data = self.msg_to_cams[target_cam_id].get()
            send_msg(self.sock_cams[target_cam_id], msg_type, msg_data) 
        
    def load_frames(self, frame_list):
        '''
            load frames in memory and put them into frame_list
        '''
        logger = create_logger()
        frame_idx = 0

        while self.video_capture.is_opened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            img_tensor = transforms.Resize(self.frame_size, antialias=True)(transforms.ToTensor()(frame))
            frame_list.put((frame_idx, img_tensor)) 
            frame_idx += 1

        logger.info(f"cam {self.cam_id} frames loaded!, total frame {frame_idx}")
    
    def run_group_frames(self, frame_list, clip_list):
        '''
            read frames from frame_list and group them into clip_list
            16 frames as a group, with 32 fps 
        '''
        logger = create_logger()
        time.sleep(50)
        logger.info("start processing!")
        while True:
            img_tensors = []
            for i in range(self.num_frames):
                frame_idx, img_tensor = frame_list.get()
                img_tensors.append(img_tensor)
            video = torch.stack(img_tensors, dim=1).unsqueeze(0) # 1, 3, num_frames, 224, 224
            logger.info(f"grouped  [src={self.cam_id},frames={frame_idx - self.num_frames + 1}-{frame_idx}]")
            clip_list.put((self.cam_id, frame_idx-self.num_frames+1, video, 0, time.time()))
            time.sleep(0.5) 
                
    def run_receiver_from_camera(self, from_cam_id, decode_list):
        '''
            Receive workload from other cameras.
        '''
        logger = create_logger()
        while True:
            msg_type, msg_data, data_size = recv_msg(self.sock_cams[from_cam_id])

            if msg_type == 1:
                # workload from other camera
                src_cam_id, frame_start_idx, encoded_video, latency = msg_data
                t_recv = time.time()
                logger.info(f'Receive from camera {from_cam_id}: [src={src_cam_id},frames={frame_start_idx}-{frame_start_idx+self.num_frames-1}]')
                bandwidth = self.bandwidths[from_cam_id] 
                logger.info(f'with bandwidth: {bandwidth:.2f} MB/s')
                latency += data_size/1024/1024 / bandwidth # accumulate network latency
                # put into decode_list
                decode_list.put([src_cam_id, frame_start_idx, encoded_video, latency, t_recv])
            else:
                logger.info(f'Unknown msg type: {msg_type}')
                raise NotImplementedError
    
    def run_receiver_from_edge_server(self):
        logger = create_logger()
        while True:
            msg_type, msg_data, data_size = recv_msg(self.sock_edge)
            if msg_type == 3:
                # receive migration signal from edge server
                dst_cam_id, migrate_amount = msg_data
                logger.info(f'Receive migration signal: {migrate_amount} to cam {dst_cam_id}')
                self.migrate_list.append((dst_cam_id, migrate_amount)) 
            else:
                logger.info(f'Unknown msg type: {msg_type}')
                raise NotImplementedError
                
    def run_main(self, clip_list):
        logger = create_logger()
        dest_cam_id, migrate_amount = None, 0 
        while True:
            src_cam_id, frame_start_idx, video, latency, t_recv = clip_list.get()
            logger.info(f'process [src={src_cam_id},frames={frame_start_idx}-{frame_start_idx+self.num_frames-1}]')

            if dest_cam_id is None:
                try:
                    dest_cam_id, migrate_amount = self.migrate_list.get(block=False)
                except queue.Empty:
                    pass
            
            if dest_cam_id is not None:
                if migrate_amount > 16:
                    logger.info(f'migrate to cam {dest_cam_id}: [src={src_cam_id},frames={frame_start_idx}-{frame_start_idx+self.num_frames-1}]')
                    migrate_amount -= 16
                    
                    # encode video with filter
                    t1 = time.time()
                    encoded_video = self.codec.encode_with_filter(video)
                    t2 = time.time()
                    logger.info(f"encode time: {t2-t1:.2f}s")

                    if args.codec == 'mesh':
                        encoded_video = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in encoded_video]
                    if args.codec == 'crucio':
                        encoded_video[0] = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in encoded_video[0]]

                    latency += time.time() - t_recv
                    logger.info(f"send to camera, time = {time.time()}")
                    msg_type = 1
                    msg_data = (src_cam_id, frame_start_idx, encoded_video, latency)
                    self.send_to_camera(dest_cam_id, msg_type, msg_data)
                else:
                    dest_cam_id = None
                    
            else:
                t1 = time.time()
                category = self.classifier.classify(video)
                t2 = time.time()
                logger.info(f"classify time: {t2-t1:.2f}s")
                logger.info(f"category = {category}")
               
                # encode video with filter
                t1 = time.time()
                encoded_video = self.codec.encode_with_filter(video)
                t2 = time.time()
                logger.info(f"encode time: {t2-t1:.2f}s")

                if args.codec == 'mesh':
                    encoded_video = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in encoded_video]
                if args.codec == 'crucio':
                    encoded_video[0] = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in encoded_video[0]]

                latency += time.time() - t_recv
                logger.info(f"send to edge server, time = {time.time()}")
                msg_type = 1
                msg_data = (src_cam_id, frame_start_idx, encoded_video, category, latency)
                self.send_to_edge(msg_type, msg_data)
     
    def run_decode(self, decode_list, clip_list):
        '''
            decode videos from other cameras
        '''
        logger = create_logger()
        while True:
            src_cam_id, frame_start_idx, encoded_video, latency, t_recv = decode_list.get()
            logger.info(f'decode [src={src_cam_id},frames={frame_start_idx}-{frame_start_idx+self.num_frames-1}]')
            t1 = time.time() 
            video, select = self.codec.decode(encoded_video)
            t2 = time.time()
            logger.info(f"decode time: {t2-t1:.2f}s")
            # select is all 1 for encoded video from other cameras
            video = video.detach().cpu()
            clip_list.put([src_cam_id, frame_start_idx, video, latency, t_recv]) 
    
    def run_report_runtime_states(self):
        logger = create_logger()
        while True:
            time.sleep(0.5)
            workload = self.get_workload()
            bandwidths = self.get_bandwidths()
            #logger.info(f'cam {self.cam_id} report: workload={workload} bandwidths={bandwidths}')
            #self.send_to_edge(2, [workload, bandwidths])
            
    def process_networking(self): 
        '''
        we use one process with multithreads to handle all networking tasks
        '''
        threads = []
        threads.append(threading.Thread(target=self.run_receiver_from_edge_server))
        threads.append(threading.Thread(target=self.run_sender_to_edge))
        for i in range(len(self.sock_cams)):
            if i != self.cam_id:
                threads.append(threading.Thread(target=self.run_receiver_from_camera, args=(i, self.decode_list)))
                threads.append(threading.Thread(target=self.run_sender_to_camera, args=(i,)))
            
        [t.start() for t in threads] 
        [t.join() for t in threads]
    
    def process_gpu(self):
        '''
        we use one process with multithreads to handle all gpu related tasks. there's only one GPU on the camera side.
        '''
        logger = create_logger()
        device = 0 
        # threads in one process share the same cuda context
        self.codec.load_model(device=device)
        self.classifier.load_model(device=device)
        logger.info("all models loaded!")
        self.codec.warmup()
        self.classifier.warmup()
        logger.info("all warm up finished!")
         
        threads = []
        threads.append(threading.Thread(target=self.run_main, args=(self.clip_list,)))
        threads.append(threading.Thread(target=self.run_decode, args=(self.decode_list, self.clip_list)))

        [t.start() for t in threads] 
        [t.join() for t in threads]

    def process_capture(self):
        self.load_frames(self.frame_list)
        self.run_group_frames(self.frame_list, self.clip_list)
        #threads = []
        #threads.append(threading.Thread(target=self.run_capture_frames, args=(self.frame_list,)))
        #threads.append(threading.Thread(target=self.run_group_frames, args=(self.frame_list, self.clip_list)))
        #[t.start() for t in threads] 
        #[t.join() for t in threads]
        
    def process_others(self):
        threads = []
        threads.append(threading.Thread(target=self.run_report_runtime_states))

        [t.start() for t in threads] 
        [t.join() for t in threads]

    def run(self):
        processes = []
        processes.append(multiprocessing.Process(target=self.process_gpu))
        processes.append(multiprocessing.Process(target=self.process_capture))
        processes.append(multiprocessing.Process(target=self.process_others))
        
        [p.start() for p in processes] 

        self.process_networking() # run in main process

        [p.join() for p in processes]


server_ip = info['edge_server_info']['ip']
server_port = info['edge_server_info']['port']

my_ip = scenario['cameras'][args.cam_id]['ip']
my_port = scenario['cameras'][args.cam_id]['port']

sock_cams = [None for _ in range(num_cameras)]
# connect to edge server
edge_server_ip = info['edge_server_info']['ip']
edge_server_port = info['edge_server_info']['port']
sock_edge = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_edge.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_edge.bind((my_ip, my_port))
logger.info(f"connect to edge server {edge_server_ip} {edge_server_port}")
sock_edge.connect((edge_server_ip, edge_server_port))

# listen for connections from cameras with lower id
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((my_ip, my_port))
listen_socket.listen(10)
logger.info(f"cam {args.cam_id} listening on ({my_ip},{my_port})...")

num_connected = 0
while num_connected < args.cam_id:
    conn, addr = listen_socket.accept()
    logger.info(f"received connection {addr}:")

    num_connected += 1
    for i in range(0, num_cameras):
        if scenario['cameras'][i]['ip'] == addr[0] and scenario['cameras'][i]['port'] == addr[1]:
            logger.info(f"cam {args.cam_id} receive connection from cam {i}")
            sock_cams[i] = conn
            break

logger.info(f"cam {args.cam_id} connected to all cameras with lower id")
listen_socket.close()

# connect to cameras with higher id
for i in range(args.cam_id + 1, num_cameras):
    target_ip = scenario['cameras'][i]['ip']
    target_port = scenario['cameras'][i]['port']
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((my_ip, my_port))
    
    sock = connect_with_retry(sock, target_ip, target_port) 
    
    logger.info(f"cam {args.cam_id} connected to cam {i} on ({target_ip},{target_port})")
    sock_cams[i] = sock

logger.info(f"all connections completed!")

# video data

video_name = scenario['cameras'][args.cam_id]['video']
video_dir = os.path.join(DATA_DISK_DIR, 'videos_dir', video_name)
video_info = info['videos_info'][video_name]
video_info['video_dir'] = video_dir

logger.info("video info:", video_info)

num_frames = 16

bandwidth_matrix = np.array(scenario['bandwidth_matrix'], dtype=np.float32)

smartcamera = Camera(cam_id=args.cam_id, num_frames=num_frames, video_info=video_info, bandwidth_matrix=bandwidth_matrix, sock_edge=sock_edge, sock_cams=sock_cams)

logger.info("start!")
smartcamera.run()
