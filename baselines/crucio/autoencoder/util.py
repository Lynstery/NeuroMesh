import glob
import locale
import lzma
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
import ffmpeg 
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
#from torch.distributed import init_process_group

# Whether CUDA GPUs are used during evaluation
CUDA_ENABLED = True

shell = subprocess.check_output(
    "env | grep SHELL=", shell=True).decode().strip()
if shell == "SHELL=/bin/bash":
    rc = ".bashrc"
else:
    rc = ".zshrc"
home = str(Path.home())
find = subprocess.check_output(
    f"grep -n 'crucio' {home}/{rc}", shell=True).decode().strip()
PROJECT_DIR = find.split(" ")[-1]

# Maximum frame number in dataset
find = subprocess.check_output(
    f"grep -n 'MAX_FRAME_NUM=' {PROJECT_DIR}/video_data/prepare.sh", shell=True).decode().strip()
MAX_FRAME_NUM = int(find.split("=")[1])

loc = locale.getlocale()[0]
username = os.getlogin()
'''
if loc == 'zh_CN':
    media = '/media/'+username+'/数据硬盘/下载'
    if os.path.isdir(media) and os.listdir(media):
        DOWNLOAD_DIR = media
    else:
        DOWNLOAD_DIR = '/media/'+username+'/PortableSSD/下载'
else:
    DOWNLOAD_DIR = '/data/'+username+'/crucio_downloads'
'''
DOWNLOAD_DIR = '/data/'+username+'/crucio_downloads'

# png is lossless image compression (captured by camera)
IMAGE_EXT = '.png'
# Whether to use YUV color space during codec
YUV_ENABLED = False
if YUV_ENABLED:
    WEIGHTS_DIR = DOWNLOAD_DIR+'/yuv_weights'
else:
    WEIGHTS_DIR = DOWNLOAD_DIR+'/rgb_weights'
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

def check_work_dir():
    '''
    Switch working directory to directory where py file (to be run) is located
    Calling check_work_dir function at beginning of py file running as main function
      corrects all relative paths in it
    Function is invalid if py file is called as a module by another main function
    In this case you should use absolute path 
      (configure project directory with command ./configure.sh 1)
    '''
    curr_dir = os.getcwd()
    file_path = sys.argv[0]
    # file_name = os.path.basename(file_path)
    # file_name = file_path.split('/')[-1]
    if file_path[0] == '/':
        work_dir = file_path
    else:
        if file_path[0] == '.' and file_path[1] == '/':
            work_dir = os.path.join(curr_dir, file_path[1:])
            # work_dir = curr_dir+file_path[1:]
        else:
            work_dir = os.path.join(curr_dir, file_path)
            # work_dir = curr_dir+'/'+file_path
    work_dir = os.path.dirname(work_dir)
    # work_dir = work_dir[:-(len(file_name))]
    if os.path.exists(work_dir):
        os.chdir(work_dir)

def load_from_encoded_video_to_tensor(video_path, probe, duration, offset, num_frames, step, random_step=False, is_yuv=YUV_ENABLED, is_gpu=CUDA_ENABLED, size=None, rank=0):
    '''
    Load multiple consecutive video frames from video file and return a tensor (no batch dimension)
    video_path -> Absolute path to video file 
    probe -> Video probe information
    duration -> Total number of frames in video
    offset -> Number of first frame
    length -> Number of frames to load
    '''
    rand_step_list = [random.randint(1, 5) for _ in range(0, num_frames)]
    frame_id_list = []
    for i in range(0, num_frames):
        frame_id = offset - 1
        frame_id_list.append(frame_id)
        cur_step = rand_step_list[i] if random_step else step
        if offset + cur_step < duration:
            offset += cur_step
    try:
        out, err = (
            ffmpeg
            .input(video_path)
            .filter('select', '+'.join(f'eq(n,{frame_id})' for frame_id in frame_id_list))
            .filter('setpts', 'N/FRAME_RATE/TB')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        height = int(probe['streams'][0]['height'])
        width = int(probe['streams'][0]['width'])
        video_data = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        if video_data.shape[0] < len(frame_id_list):
            append_num = len(frame_id_list) - video_data.shape[0]
            last_frame = video_data[-1][np.newaxis, :]
            video_data = np.append(video_data, np.repeat(last_frame, append_num, axis=0), axis=0)
        assert video_data.shape[0] == len(frame_id_list)
        sampled_list = [Image.fromarray(frame).convert('RGB') for frame in video_data]
    except Exception:
        raise RuntimeError(f'Error occurred in reading frames {frame_id_list} from video {video_path} of duration {duration}. FFmpeg: {str(err)}')

    imgs = [] 
    for image in sampled_list:
        img = convert_loaded_image_to_tensor(image, is_yuv=is_yuv, is_gpu=is_gpu, size=size, rank=rank) 
        imgs.append(img)
    tensor = torch.stack(imgs, dim=1)

    return tensor

def convert_loaded_image_to_tensor(img, is_yuv, is_gpu, size=None, rank=0):
    if is_yuv:
        tensor_mode = 'YCbCr'
    else:
        tensor_mode = 'RGB'
    image = img.convert(tensor_mode)
    tensor = transforms.ToTensor()(image)
    if is_gpu:
        tensor = tensor.to(rank)
    if size:
        tensor = transforms.Resize(size, antialias=True)(tensor)
    return tensor


def convert_image_to_tensor(img_path, is_yuv, is_gpu, size=None, rank=0):
    '''
    Convert an RGB image to a tensor (no batch dimension)
    img_path -> Image absolute path 
    is_yuv -> Whether to use YUV color space
    is_gpu -> Whether to move tensor to GPU
    size -> List [H,W] for image resize
    '''
    if is_yuv:
        tensor_mode = 'YCbCr'
    else:
        tensor_mode = 'RGB'
    image = Image.open(img_path).convert(tensor_mode)
    tensor = transforms.ToTensor()(image)
    if is_gpu:
        tensor = tensor.to(rank)
    if size:
        tensor = transforms.Resize(size, antialias=True)(tensor)
    return tensor


def convert_image_colorspace(tensor, rgb_to_yuv=True):
    '''
    Change color space of image tensor
    tensor -> A batch of images
    rgb_to_yuv -> RGB to YCbCr (True) or YCbCr to RGB (False)

    https://github.com/python-pillow/Pillow/blob/main/src/libImaging/ConvertYCbCr.c

    Y  = R *  0.29900 + G *  0.58700 + B *  0.11400
    Cb = R * -0.16874 + G * -0.33126 + B *  0.50000 + 128
    Cr = R *  0.50000 + G * -0.41869 + B * -0.08131 + 128

    R  = Y +                       + (Cr - 128) *  1.40200
    G  = Y + (Cb - 128) * -0.34414 + (Cr - 128) * -0.71414
    B  = Y + (Cb - 128) *  1.77200
    '''
    converted_tensor = torch.zeros_like(tensor)
    # Y or R
    channel1 = tensor[:, 0, :, :]
    # Cb or G
    channel2 = tensor[:, 1, :, :]
    # Cr or B
    channel3 = tensor[:, 2, :, :]
    if rgb_to_yuv:
        add = 128/255
        sub = 0
        matrix = torch.tensor([[0.29900, 0.58700, 0.11400],
                               [-0.16874, -0.33126, 0.50000],
                               [0.50000, -0.41869, -0.08131]])
    else:
        add = 0
        sub = -128/255
        matrix = torch.tensor([[1, 0, 1.40200],
                               [1, -0.34414, -0.71414],
                               [1, 1.77200, 0]])
    converted_tensor[:, 0, :, :] = matrix[0, 0] * channel1 + \
        matrix[0, 1] * (channel2 + sub) + matrix[0, 2] * (channel3 + sub)
    converted_tensor[:, 1, :, :] = add + matrix[1, 0] * channel1 + \
        matrix[1, 1] * (channel2 + sub) + matrix[1, 2] * (channel3 + sub)
    converted_tensor[:, 2, :, :] = add + matrix[2, 0] * channel1 + \
        matrix[2, 1] * (channel2 + sub) + matrix[2, 2] * (channel3 + sub)
    converted_tensor = torch.clamp(converted_tensor, 0, 1)
    return converted_tensor


class ColorConverter(nn.Module):
    '''
    Due to floating-point arithmetic and value rounding,
      color space conversions inevitably lead to error accumulation
    YUV is not recommended unless extremely high compression ratios are required
    '''

    def __init__(self):
        super(ColorConverter, self).__init__()

    def forward(self, x):
        if YUV_ENABLED:
            return convert_image_colorspace(x, False)
        else:
            return x


def convert_tensor_to_image(tensor, is_yuv):
    '''
    Converts a tensor (no batch dimension) to an RGB image
    tensor -> Usually output tensor from decoder 
    is_yuv -> Whether tensor is in YUV color space
    '''
    if is_yuv:
        tensor = convert_image_colorspace(tensor.unsqueeze(0), False)[0]
    image = transforms.ToPILImage(mode='RGB')(tensor.detach().cpu())
    return image


def save_compressed_data(input_path, compressed_data):
    '''
    input_path -> Absolute path of input image or video
    compressed_data -> Compressed data from encoder (tensor)
    Return absolute path to compressed data
    '''
    compressed_data = lzma.compress(pickle.dumps(compressed_data))
    data_name, ext = os.path.splitext(input_path)
    data_path = data_name + ".pkl"
    with open(data_path, 'wb') as f:
        f.write(compressed_data)
    return data_path


def save_compressed_data_at(data_path, compressed_data):
    '''
    data_path -> Absolute path to save compressed data
    compressed_data -> Compressed data from encoder (tensor)
    Return absolute path to compressed data
    '''
    compressed_data = lzma.compress(pickle.dumps(compressed_data))
    with open(data_path, 'wb') as f:
        f.write(compressed_data)
    return data_path

def load_compressed_data(data_path, rank=None):
    '''
    data_path -> Absolute path of compressed data
    Return compressed data from encoder (tensor)
    '''
    with open(data_path, 'rb') as f:
        compressed_data = f.read()
    compressed_data = pickle.loads(lzma.decompress(compressed_data))
    if rank is not None:
        compressed_data = compressed_data.to(rank)
    return compressed_data


def get_folder_size(folder_path):
    '''
    Calculates total size of all files in specified folder (KB)
    '''
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size/1024


def load_video_to_tensor(video_path, start_num=1, length=MAX_FRAME_NUM, return_list=False,
                         check=False, is_yuv=YUV_ENABLED, is_gpu=CUDA_ENABLED, resize=None):
    '''
    Loads multiple consecutive video frames and returns a tensor (no batch dimension)
    video_path -> Absolute path to video directory
    start_num -> Number of first frame (starting from 1)
    length -> Number of frames to load
    return_list -> Whether to return a list of tensors
    check -> True only when checking dataset
    '''
    assert os.path.exists(video_path)
    assert start_num >= 1
    if check:
        files = glob.glob(os.path.join(video_path, '*'+IMAGE_EXT))
        file_name = os.path.basename(files[0])
        string = os.path.splitext(file_name)[0]
        assert re.match(r'frame\d{4}', string)
        start_num = 1
        length = len(files)
    imgs = []
    curr_length = 0
    curr_num = start_num
    while curr_length < length:
        if curr_num > MAX_FRAME_NUM:
            break
        str_num = "{:04d}".format(curr_num)
        img_path = os.path.join(video_path, 'frame'+str_num+IMAGE_EXT)
        if os.path.exists(img_path):
            img = convert_image_to_tensor(img_path, is_yuv, is_gpu, resize)
            imgs.append(img)
            curr_length += 1
        curr_num += 1
    if return_list:
        tensor = imgs, curr_num
    else:
        tensor = torch.stack(imgs, dim=1)
    return tensor


def save_tensor_to_video(video_path, tensor, is_yuv):
    '''
    Saves a tensor (no batch dimension) to specified video directory
    video_path -> Absolute path to video directory
    tensor -> A tensor from decoder
    is_yuv -> Whether tensor is in YUV color space
    '''
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    len = tensor.shape[1]
    for _ in range(len):
        img = convert_tensor_to_image(tensor[:, _], is_yuv)
        str_num = "{:04d}".format(_+1)
        img.save(os.path.join(video_path, 'frame'+str_num+IMAGE_EXT))
