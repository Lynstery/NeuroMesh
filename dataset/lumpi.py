import os

import ffmpeg
import numpy as np
import torch
from PIL import Image

from mesh.dataset.utils import (LUMPI_DIR, TransformForLUMPIDataset)
from mesh.dnn_model.util import (convert_a_video_tensor_to_images)

class MultiViewDataset(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    """

    def __init__(self,
                 mode='train',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 load_ratio=1.0):

        super(MultiViewDataset, self).__init__()
        self.mode = mode
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.transform = transform

        self.clips_with_ref = self._make_dataset(load_ratio)
        print(f"Dataset loaded with {len(self.clips_with_ref)} clips")

    def __getitem__(self, index):

        video_name, ref_video_name = self.clips_with_ref[index]

        #print(f"{video_name}, {ref_video_name}")
        probe = ffmpeg.probe(video_name)
        duration = int(probe['streams'][0]['nb_frames'])

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_ffmpeg_batch_loader(
            video_name, probe, duration, segment_indices, skip_offsets)
        images = self.transform((images, None))  # T*C,H,W
        
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        images = images.view((self.new_length, 3) + images.size()[-2:]).transpose(0, 1)

        ref_probe = ffmpeg.probe(ref_video_name)
        ref_duration = int(probe['streams'][0]['nb_frames'])
        ref_images = self._video_TSN_ffmpeg_batch_loader(
            ref_video_name, ref_probe, ref_duration, segment_indices, skip_offsets)
        ref_images = self.transform((ref_images, None))
        ref_images = ref_images.view((self.new_length, 3) + ref_images.size()[-2:]).transpose(0, 1)

        return (images, ref_images)

    def __len__(self):
        return len(self.clips_with_ref)

    def _make_dataset(self, load_ratio=1.0):
        csv_path = LUMPI_DIR + '/clips_pair.csv' 
        clips_with_ref = []
        with open(csv_path, 'r') as f_csv:
            for line in f_csv.readlines():
                line = line.strip() 
                images, ref_images = line.split(' ')[0], line.split(' ')[1]
                clips_with_ref.append((images, ref_images))
        
        clips_with_ref = clips_with_ref[:int(len(clips_with_ref) * load_ratio)]
        return clips_with_ref

    def _sample_train_indices(self, num_frames):
        average_duration = (
            num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_ffmpeg_batch_loader(self, video_name, probe, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        try:
            out, err = (
                ffmpeg
                .input(video_name)
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
            raise RuntimeError(f'Error occurred in reading frames {frame_id_list} from video {video_name} of duration {duration}. FFmpeg: {str(err)}')
        return sampled_list


def build_lumpi_multiview_dataset(args, load_ratio=0.8):
    dataset = MultiViewDataset(
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=TransformForLUMPIDataset(args),
        temporal_jitter=False,
        load_ratio=load_ratio)
    return dataset

import matplotlib.pyplot as plt
from mesh.dnn_model.util import convert_tensor_to_an_image
import argparse
from mesh.dataset.utils import timm_video_normalization
from mesh.vitransformer.arguments import get_args
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_args()

    patch_size = (16, 16) 
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset = build_lumpi_multiview_dataset(args, load_ratio=1)

    inputs, ref_inputs = dataset[0]
    print(f"inputs.shape = {inputs.shape}")
    print(f"ref_inputs.shape = {ref_inputs.shape}")

    inputs = inputs.unsqueeze(0)
    ref_inputs = ref_inputs.unsqueeze(0)
    inputs = timm_video_normalization(inputs, unnorm=True)
    ref_inputs = timm_video_normalization(ref_inputs, unnorm=True)
    inputs = inputs.squeeze(0)
    ref_inputs = ref_inputs.squeeze(0)

    for i in range(0, args.num_frames):
        img_0 = convert_a_video_tensor_to_images(inputs, i, False)
        img_1 = convert_a_video_tensor_to_images(ref_inputs, i, False)
        img_0.save(f"/data/zh/test/img_{i}.png")
        img_1.save(f"/data/zh/test/img_ref_{i}.png")
