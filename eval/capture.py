import os
import time
from PIL import Image
import cv2
import os
import numpy as np
import queue
import torch
import struct
import lzma
import pickle
from mesh.dnn_model.util import convert_pil_image_to_tensor

class ImageFolderVideoCapture:
    def __init__(self, folder_path, frame_rate, frame_total=None, img_ext='.png'):
        self.folder_path = folder_path
        self.frame_rate = frame_rate
        self.img_ext = img_ext
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(img_ext)])
        self.current_frame_index = 0

        print(f"found {len(self.image_files)} frames")
        self.frame_total = len(self.image_files) 
        if self.frame_total == 0:
            raise ValueError("No images found in the folder.")

        if frame_total is not None:
            self.frame_total = min(frame_total, self.frame_total)
        
        first_image = Image.open(os.path.join(folder_path, self.image_files[0]))
        self.frame_width, self.frame_height = first_image.size

    def read(self):
        if self.current_frame_index >= self.frame_total:
            return False, None
        
        image_file = self.image_files[self.current_frame_index]
        image_path = os.path.join(self.folder_path, image_file)
        img = Image.open(image_path)
        
        if img is None:
            return False, None

        self.current_frame_index += 1
        return True, img

    def is_opened(self):
        return self.current_frame_index < self.frame_total

    def info(self):
        return self.frame_total, self.frame_rate, self.frame_width, self.frame_height
    
    
    

        



