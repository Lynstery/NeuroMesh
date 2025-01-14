import os
import csv
import json
import torch
import pickle
import argparse
import lzma
from mesh.dnn_model.util import load_compressed_data, convert_image_to_tensor
from mesh.eval.utils import check_work_dir
from mesh.dataset.utils import DATA_DISK_DIR
from mesh.dnn_model.task import run_inference, visualize_result, calculate_accuracy

check_work_dir()

rank = 3

argparser = argparse.ArgumentParser()
argparser.add_argument('--scenario', type=str, required=True)
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

output_dir_name = f"{args.scenario}_{args.codec}_{args.load_balancer}_{args.propagate}"

OUTPUT_DIR = os.path.join(DATA_DISK_DIR, "mesh_results/", output_dir_name)

assert os.path.exists(OUTPUT_DIR)

INFER_RESULT_META_PATH = os.path.join(OUTPUT_DIR, 'meta.csv')
INFER_RESULT_PATH = os.path.join(OUTPUT_DIR, 'infer_results')

PROCESSED_INFER_RESULT_META_PATH = os.path.join(OUTPUT_DIR, 'processed_meta.csv')
STATISTICS_PATH = os.path.join(OUTPUT_DIR, 'statistics.csv')
GT_INFER_RESULT_IMG_DIR = os.path.join(OUTPUT_DIR, 'gt_results_img')

if os.path.exists(STATISTICS_PATH):
   os.remove(STATISTICS_PATH) 

with open(STATISTICS_PATH, 'w') as f:
    f.write(f'avg_acc,avg_latency,avg_throughput\n')

if os.path.exists(PROCESSED_INFER_RESULT_META_PATH):
   os.remove(PROCESSED_INFER_RESULT_META_PATH) 

with open(PROCESSED_INFER_RESULT_META_PATH, 'w') as f:
    f.write(f'src,frame_idx,select,latency,acc,current_throughput\n') 

if not os.path.exists(GT_INFER_RESULT_IMG_DIR):
    os.makedirs(GT_INFER_RESULT_IMG_DIR)
else:
    os.system(f'rm -rf {GT_INFER_RESULT_IMG_DIR}/*')

GT_VIDEOS_DIR = os.path.join(DATA_DISK_DIR, 'videos_dir')

def get_gt_img_locally(src_cam, frame_idx):
    video_name = scenario["cameras"][src_cam]["video"]
    img_path = os.path.join(GT_VIDEOS_DIR, video_name, f"{(frame_idx+1):06d}.png")
    img = convert_image_to_tensor(img_path, is_yuv=False, is_gpu=False, size=(224, 224))
    return img

def process_results():
    cnt_frames = 0 
    avg_accuracy = 0
    avg_latency = 0
    avg_throughput = 0
    with open(INFER_RESULT_META_PATH, newline='', encoding='utf-8') as csvfile:
        csv_dict_reader = csv.DictReader(csvfile)
        for row in csv_dict_reader:
            src_cam_id = int(row["src"])
            frame_idx = int(row["frame_idx"])
            H = int(row["h"])
            W = int(row["w"])
            is_selected = int(row["select"])
            latency = float(row["latency"])
            cls = int(row["category"])
            task_name = row["task_name"]
            result_save_path = row["result_save_path"]
            current_throughput = float(row["current_throughput"])
            
            with open(result_save_path, 'rb') as f:
                compressed_data = f.read()

            result = pickle.loads(lzma.decompress(compressed_data))

            if isinstance(result, torch.Tensor):
                result = result.to(rank)
            elif isinstance(result, list):
                result = [r.to(rank) for r in result]
            elif isinstance(result, dict):
                result = {k: v.to(rank) for k, v in result.items()}
             
            gt_img_tensor = get_gt_img_locally(src_cam_id, frame_idx) 
            
            gt_img_tensor = gt_img_tensor.to(rank)
            gt_result = run_inference(task_name, gt_img_tensor.unsqueeze(0), rank=rank)[0]
            
            acc = calculate_accuracy(task_name, [result], [gt_result], [H, W], gt_img_tensor.shape[-2:]) 
            video_name = scenario["cameras"][src_cam_id]["video"]
            print(f"cam{src_cam_id}, frame{frame_idx}, {task_name}, acc: {acc:.3f}") 
            '''
            valid_categories = info["videos_info"][video_name]['category'] 
            if not cls in valid_categories:
                print("wrong category, set acc = 0")
                acc = 0
            '''
            
            cnt_frames += 1 
            avg_accuracy += acc
            avg_latency += latency
            avg_throughput += current_throughput

            with open(PROCESSED_INFER_RESULT_META_PATH, 'a') as f:
                f.write(f'{src_cam_id},{frame_idx},{is_selected},{latency},{acc:.3f},{current_throughput:.3f}\n') 

            # save gt inference imgs for visualization
            visualize_result(task_name, gt_img_tensor, gt_result, os.path.join(GT_INFER_RESULT_IMG_DIR, f"{src_cam_id}_{frame_idx}_{task_name}_gt.png"))

        avg_accuracy /= cnt_frames
        avg_latency /= cnt_frames
        avg_throughput /= cnt_frames

        with open(STATISTICS_PATH, 'w') as f:
            f.write(f'{avg_accuracy},{avg_latency},{avg_throughput}\n')

process_results()