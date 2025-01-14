import os
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from mesh.dataset.masking import (EmptyMask, RandomMaskingGenerator,
                                  TubeMaskingGenerator)
from mesh.dataset.transforms import (GroupMultiScaleCrop, GroupNormalize, GroupScale,
                                     Stack, ToTorchFormatTensor)

shell = subprocess.check_output(
    "env | grep SHELL=", shell=True).decode().strip()
if shell == "SHELL=/bin/bash":
    rc = ".bashrc"
else:
    rc = ".zshrc"
home = str(Path.home())
find = subprocess.check_output(
    f"grep -n 'mesh' {home}/{rc}", shell=True).decode().strip()
PROJECT_DIR = find.split(" ")[-1]

username = os.getlogin()
if 'SSH_CLIENT' in os.environ or 'SSH_CONNECTION' in os.environ:
    DATA_DISK_DIR = '/data/'+username
else:
    MSI = '/media/'+username+'/PortableSSD'
    if os.path.isdir(MSI) and os.listdir(MSI):
        DATA_DISK_DIR = MSI
    else:
        DATA_DISK_DIR = '/media/'+username+'/数据硬盘'
KINETICS_DIR = DATA_DISK_DIR+'/kinetics-dataset'
KINETICS400_DIR = KINETICS_DIR+'/k400'
LUMPI_DIR = DATA_DISK_DIR+'/LUMPI-dataset'
MODEL_SAVE_DIR = DATA_DISK_DIR+'/mesh_model'
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)


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

class TransformForLUMPIDataset(object):
    def __init__(self, args):
        self.input_mean = list(IMAGENET_DEFAULT_MEAN)
        self.input_std = list(IMAGENET_DEFAULT_STD)
        self.transform = transforms.Compose([
            GroupScale(args.input_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(self.input_mean, self.input_std)
        ])

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = list(IMAGENET_DEFAULT_MEAN)
        self.input_std = list(IMAGENET_DEFAULT_STD)
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(
            args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        if args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        if args.mask_type == 'learnable':
            self.masked_position_generator = EmptyMask(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator)
        repr += ")"
        return repr


def timm_video_normalization(videos, unnorm=False):
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(
        videos.device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(
        videos.device)[None, :, None, None, None]
    if unnorm:
        output_videos = videos * std + mean  # in [0, 1]
    else:
        output_videos = (videos - mean) / std
    return output_videos


class ReshapeVideo(nn.Module):
    def __init__(self, num_frames):
        super(ReshapeVideo, self).__init__()
        self.frame_num = num_frames

    def forward(self, x):
        shape_size = len(x.shape)
        assert shape_size == 4 or shape_size == 5
        if shape_size == 4:
            x = x.reshape(-1, self.frame_num,
                          x.shape[-3], x.shape[-2], x.shape[-1])
            return x.permute(0, 2, 1, 3, 4)
        elif shape_size == 5:
            self.frame_num = x.shape[2]
            x = x.permute(0, 2, 1, 3, 4)
            return x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])


def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV image."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def random_perspective_transform(pil_image):
    """Apply random perspective transform to a PIL image."""
    width, height = pil_image.width, pil_image.height

    # Define source points
    src_points = [(0, 0), (width - 1, 0),
                  (width - 1, height - 1), (0, height - 1)]

    # Define destination points with random perturbations
    dst_points = [(random.randint(0, width // 4), random.randint(0, height // 4)),
                  (random.randint(3 * width // 4, width),
                   random.randint(0, height // 4)),
                  (random.randint(3 * width // 4, width),
                   random.randint(3 * height // 4, height)),
                  (random.randint(0, width // 4), random.randint(3 * height // 4, height))]

    # Convert PIL image to OpenCV image
    cv2_image = pil_to_cv2(pil_image)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(
        np.float32(src_points), np.float32(dst_points))

    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(
        cv2_image, matrix, (cv2_image.shape[1], cv2_image.shape[0]))

    # Convert back to PIL image
    return cv2_to_pil(transformed_image)


if __name__ == '__main__':
    print(KINETICS_DIR)
