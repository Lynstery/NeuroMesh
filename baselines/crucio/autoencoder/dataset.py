import glob
import os
import re
import subprocess
import time
from multiprocessing.pool import ThreadPool
import csv
import cv2
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from crucio.autoencoder.util import (DOWNLOAD_DIR, IMAGE_EXT, MAX_FRAME_NUM,
                                     PROJECT_DIR, YUV_ENABLED,
                                     convert_image_to_tensor,
                                     load_video_to_tensor,
                                     load_from_encoded_video_to_tensor)

# https://cocodataset.org/#download
IMAGE_DIR = DOWNLOAD_DIR+'/val2017'
# https://data.vision.ee.ethz.ch/cvl/youtube-objects/
VIDEO_DIR = DOWNLOAD_DIR+'/youtube-objects'

KINETICS400_DIR = "/data/zh/kinetics-dataset/k400"


# Resolution must be multiple of max stride 32 (YOLOv5)
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 640
# Minimum frame number supported by 3D-CNN structure
MIN_FRAME_NUM = 5
# Frame number used when training 3D-CNN Encoder/Decoder
CNN_FRAME_NUM = 5
assert CNN_FRAME_NUM >= MIN_FRAME_NUM
# Frame number used when training GRU Encoder
GRU_FRAME_NUM = 8
assert GRU_FRAME_NUM >= CNN_FRAME_NUM


def reduce_resolution_under_limited_memory(reduce_step, lr_dir=None):
    '''
    Excessive reduction of resolution during training
      significantly affects evaluation performance
    '''
    resize = None
    lr_resize = None
    assert isinstance(reduce_step, int)
    if reduce_step > 0:
        new_height = IMAGE_HEIGHT-32*reduce_step
        new_width = IMAGE_WIDTH-32 * int(reduce_step*IMAGE_WIDTH/IMAGE_HEIGHT)
        resize = [new_height, new_width]
        if lr_dir:
            match = re.search(r'lrx(\d+)', lr_dir)
            if match:
                scale_ratio = int(match.group(1))
            lr_resize = [int(new_height/scale_ratio),
                         int(new_width/scale_ratio)]
    return resize, lr_resize


class ImageDataset(Dataset):
    def __init__(self, root_dir, sampler_rate=1, reduce_step=0, lr_dir=None):
        assert os.path.exists(root_dir)
        image_list = glob.glob(os.path.join(root_dir, '*[0-9]'+IMAGE_EXT))
        assert sampler_rate <= 1
        if sampler_rate < 1:
            self.image_list = []
            for i in range(int(sampler_rate*len(image_list))):
                self.image_list.append(image_list[i])
        else:
            self.image_list = image_list
        print(f'Original image resolution is {IMAGE_HEIGHT}*{IMAGE_WIDTH}')
        self.resize, self.lr_resize = reduce_resolution_under_limited_memory(
            reduce_step, lr_dir)
        self.lr_dir = lr_dir
        if self.lr_dir:
            assert os.path.exists(self.lr_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # You should generate batches on CPU to save GPU memory, and move
        # them to GPU during training (according to rank in DDP mode)
        img_name = self.image_list[idx]
        image = convert_image_to_tensor(img_name, False, False, self.resize)
        if self.lr_dir:
            lr_image_name = self.lr_dir+'/'+os.path.basename(img_name)
            lr_image = convert_image_to_tensor(
                lr_image_name, YUV_ENABLED, False, self.lr_resize)
            return image, lr_image, img_name
        else:
            input = convert_image_to_tensor(
                img_name, YUV_ENABLED, False, self.resize)
            return image, input, img_name


def resize_video_for_limited_memory_and_downsample(reduce_step, scale_ratio):
    '''
    Excessive reduction of resolution during training
      significantly affects evaluation performance
    '''
    resize = None
    lr_resize = None
    assert isinstance(reduce_step, int)
    new_height = IMAGE_HEIGHT
    new_width = IMAGE_WIDTH
    resize = [new_height, new_width]
    lr_resize = [new_height, new_width]
    if reduce_step > 0:
        new_height = IMAGE_HEIGHT-32*reduce_step
        new_width = IMAGE_WIDTH-32 * int(reduce_step*IMAGE_WIDTH/IMAGE_HEIGHT)
        resize = [new_height, new_width]
    if scale_ratio > 1:
        lr_resize = [int(new_height/scale_ratio), int(new_width/scale_ratio)]
    else:
        lr_resize = resize
    return resize, lr_resize

class KineticsVideoDataset(Dataset):
    def __init__(self, mode, frame_num, sampler_rate=1.0, reduce_step=0, scale_ratio=1, frame_skip_step=5, frame_random_skip_step=False):
        assert os.path.exists(KINETICS400_DIR)
        assert mode in ['train', 'test', 'val']
        self.mode = mode

        self.labels_400 = {} # class_name -> class_id
        with open(KINETICS400_DIR+"/kinetics_400_labels.csv") as f:
            reader_labels = csv.reader(f, delimiter="\t")
            next(reader_labels)
            for i, line in enumerate(reader_labels):
                n = line[0].split(",")[1]
                v = line[0].split(",")[0]
                self.labels_400[n] = v

        self.video_list = [] # (video_path, duration, class_id)
        with open(KINETICS400_DIR+f"/{mode}.csv") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                video_path, duration, class_id = line[0].split(" ")
                self.video_list.append([video_path, int(duration), int(class_id)])

        assert sampler_rate <= 1
        if sampler_rate < 1:
            sampled_len = int(sampler_rate * len(self.video_list))
            indices = np.random.choice(len(self.video_list), sampled_len, replace=False)
            self.video_list = [self.video_list[i] for i in indices]

        self.step = frame_skip_step
        self.num_frames = frame_num
        self.random_step = frame_random_skip_step

        assert isinstance(scale_ratio, int)
        self.resize, self.lr_resize = resize_video_for_limited_memory_and_downsample(
            reduce_step, scale_ratio)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, duration, class_id = self.video_list[idx]
        duration = duration*60
        # we want the original video path for better resolution
        # video_path could be : /data/zh/kinetics-dataset/k400/k400_320p/-0SUZKVInwE_000081_000091.mp4
        # change to /data/zh/kinetics-dataset/k400/{self.mode}/-0SUZKVInwE_000081_000091.mp4 
        video_path = KINETICS400_DIR + f"/{self.mode}/" + os.path.basename(video_path)

        probe = ffmpeg.probe(video_path)
        video = load_from_encoded_video_to_tensor(
            video_path=video_path, probe=probe, duration=duration, offset=1, num_frames=self.num_frames, step=self.step, random_step=self.random_step, is_yuv=False, is_gpu=False, size=self.resize)
        input = load_from_encoded_video_to_tensor(
            video_path=video_path, probe=probe, duration=duration, offset=1, num_frames=self.num_frames, step=self.step, random_step=self.random_step, is_yuv=YUV_ENABLED, is_gpu=False, size=self.lr_resize)

        return video, input, video_path

class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_num, sampler_rate=1, reduce_step=0, scale_ratio=1):
        assert os.path.exists(root_dir)
        self.frame_num = frame_num
        video_list = glob.glob(os.path.join(root_dir, "*[0-9]"))
        self.video_list = []
        self.video_frame_list = []
        sampler_len = 0
        assert sampler_rate <= 1
        for v in range(len(video_list)):
            video = video_list[v]
            if os.path.isdir(video):
                imgs = glob.glob(os.path.join(video, '*'+IMAGE_EXT))
                if MAX_FRAME_NUM == len(imgs):
                    if sampler_len < int(sampler_rate*len(video_list)):
                        self.video_list.append(video)
                        files = []
                        for _ in range(self.frame_num):
                            str_num = "{:04d}".format(_+1)
                            files.append(os.path.join(
                                video, 'frame'+str_num+IMAGE_EXT))
                        self.video_frame_list.append(files)
                        sampler_len += 1
        print(f'Original image resolution is {IMAGE_HEIGHT}*{IMAGE_WIDTH}')
        assert isinstance(scale_ratio, int)
        self.resize, self.lr_resize = resize_video_for_limited_memory_and_downsample(
            reduce_step, scale_ratio)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        # You should generate batches on CPU to save GPU memory, and move
        # them to GPU during training (according to rank in DDP mode)
        video_path = self.video_list[idx]
        video = load_video_to_tensor(
            video_path, length=self.frame_num, is_yuv=False, is_gpu=False, resize=self.resize)
        input = load_video_to_tensor(
            video_path, length=self.frame_num, is_yuv=YUV_ENABLED, is_gpu=False, resize=self.lr_resize)
        files = self.video_frame_list[idx]
        return video, input, files


class ReshapeVideo(nn.Module):
    def __init__(self):
        super(ReshapeVideo, self).__init__()
        self.frame_num = CNN_FRAME_NUM

    def forward(self, x):
        shape_size = len(x.shape)
        assert shape_size == 4 or shape_size == 5
        if shape_size == 4:
            return x.reshape(-1, x.shape[-3], self.frame_num, x.shape[-2], x.shape[-1])
        elif shape_size == 5:
            self.frame_num = x.shape[2]
            return x.reshape(-1, x.shape[1], x.shape[3], x.shape[4])


def show_videos_difference(frame_number, video_path, reconstructed_path):
    '''
    Show image difference between original video and reconstructed video
    frame_number -> Number of frames in input video
    video_path -> Absolute path to original video directory
    reconstructed_path -> Absolute path to reconstructed video directory
    '''
    # Create a new image window
    fig = plt.figure(figsize=(12, 6))

    # Add subgraph for first row and set title
    for i in range(1, frame_number+1):
        str_num = "{:04d}".format(i)
        img_path = os.path.join(video_path, 'frame'+str_num + IMAGE_EXT)
        ax = fig.add_subplot(2, frame_number, i)
        ax.set_title('Frame'+str(i))
        ax.imshow(plt.imread(img_path))

    # Add subgraph for second row and set title
    for i in range(1, frame_number+1):
        str_num = "{:04d}".format(i)
        img_path = os.path.join(
            reconstructed_path, 'frame'+str_num + IMAGE_EXT)
        if os.path.exists(img_path):
            ax = fig.add_subplot(2, frame_number, frame_number + i)
            ax.set_title('Rec'+str(i))
            ax.imshow(plt.imread(img_path))

    # Adjust spacing between subgraphs
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    # Show images
    plt.show()


def convert_image(filename):
    format = IMAGE_EXT[1:].upper()
    try:
        img = Image.open(filename)
        # If image is already required size and format, processing is skipped
        if img.format == format and img.size == (IMAGE_WIDTH, IMAGE_HEIGHT):
            img.close()
            return

        # Resize image and save to desired format
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        new_filename = os.path.splitext(filename)[0] + IMAGE_EXT
        img.save(new_filename, format)
        img.close()

        # Delete original file
        if not filename.endswith(IMAGE_EXT):
            os.remove(filename)

    except (OSError, UnidentifiedImageError):
        # Delete other files or invalid images
        os.remove(filename)


def preprocess_image_dataset(input_dir):
    start = time.time()
    if os.path.exists(input_dir):
        # Gets filenames of all images in input directory
        image_filenames = [os.path.join(input_dir, file)
                           for file in os.listdir(input_dir)]
        frame_number = len(image_filenames)
        # Use thread pools to process all images in parallel
        pool = ThreadPool()
        pool.map(convert_image, image_filenames)
        pool.close()
        pool.join()
        print(f'Preprocessing of {frame_number} images completed')
    else:
        print(f'Directory {input_dir} does not exist')
    end = time.time()
    print(f'Running time {(end-start)/60}m')


def preprocess_video_dataset(input_dir):
    start = time.time()
    subprocess.call(['bash', PROJECT_DIR+'/video_data/prepare.sh'])
    if os.path.exists(input_dir):
        video_filenames = []
        image_filenames = []
        for f in os.listdir(input_dir):
            video_filename = os.path.join(input_dir, f)
            if os.path.isdir(video_filename):
                video_filenames.append(video_filename)
                image_filenames += [os.path.join(video_filename, file)
                                    for file in os.listdir(video_filename)]
        video_number = len(video_filenames)
        pool = ThreadPool()
        pool.map(convert_image, image_filenames)
        pool.close()
        pool.join()
        print(f'Preprocessing of {video_number} videos completed')
    else:
        print(f'Directory {input_dir} does not exist')
    end = time.time()
    print(f'Running time {(end-start)/60}m')


def check_video_dataset():
    video_list = glob.glob(os.path.join(VIDEO_DIR, "*[0-9]"))
    index = 1
    for video_name in video_list:
        tensor = load_video_to_tensor(video_name, check=True)
        assert tensor.shape == torch.Size(
            [3, MAX_FRAME_NUM, IMAGE_HEIGHT, IMAGE_WIDTH])
        print(f'num {index} path {video_name} tensor {tensor.shape}')
        index += 1


if __name__ == '__main__':
    #preprocess_image_dataset(IMAGE_DIR)
    #preprocess_video_dataset(VIDEO_DIR)
    #check_video_dataset()
    dataset = KineticsVideoDataset(mode='train', frame_num=16, sampler_rate=0.5, frame_skip_step=10)
    video, input, file_path = dataset[30]
    print(video.shape, input.shape, file_path)
    #frames = input.permute(1, 0, 2, 3)
    #for i in range(frames.shape[0]):
    #    plt.imshow(frames[i].permute(1, 2, 0))
    #    plt.savefig(f'frame_{i}.png')
    
    
