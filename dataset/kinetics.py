import os

import ffmpeg
import numpy as np
import torch
from PIL import Image

from mesh.dataset.utils import (KINETICS400_DIR, DataAugmentationForVideoMAE,
                                random_perspective_transform)


class VideoMAE(torch.utils.data.Dataset):
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
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 lazy_init=False,
                 load_ratio=1.0,
                 spatial_refer=False):

        super(VideoMAE, self).__init__()
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.transform = transform
        self.lazy_init = lazy_init
        self.spatial_refer = spatial_refer

        if not self.lazy_init:
            self.clips = self._make_dataset(setting, load_ratio)
            if len(self.clips) == 0:
                raise (RuntimeError("Found 0 video clips in subfolders of: " + KINETICS400_DIR + "\n"
                                    "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):

        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            try:
                probe = ffmpeg.probe(video_name)
                duration = int(probe['streams'][0]['nb_frames'])
            except Exception as e:
                print(
                    f"Error occurred while processing video '{video_name}': {e}\n")

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_ffmpeg_batch_loader(
            video_name, probe, duration, segment_indices, skip_offsets)
        process_data, mask = self.transform((images, None))  # T*C,H,W
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view(
            (self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)

        if self.spatial_refer is True:
            referimgs = []
            for img in images:
                referimg = random_perspective_transform(img)
                referimgs.append(referimg)
            referimgs, _ = self.transform((referimgs, None))
            referimgs = referimgs.view(
                (self.new_length, 3) + referimgs.size()[-2:]).transpose(0, 1)
            return (process_data, referimgs)
        else:
            return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, setting, load_ratio=1.0):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            total_lines = len(data)
            lines_to_load = int(total_lines * load_ratio)
            print("Loading %d videos" % lines_to_load)
            for line in data[:lines_to_load]:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_lbl_num
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips

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
            video_data = np.frombuffer(
                out, np.uint8).reshape([-1, height, width, 3])
            if video_data.shape[0] < len(frame_id_list):
                append_num = len(frame_id_list) - video_data.shape[0]
                last_frame = video_data[-1][np.newaxis, :]
                video_data = np.append(video_data, np.repeat(
                    last_frame, append_num, axis=0), axis=0)
            assert video_data.shape[0] == len(frame_id_list)
            sampled_list = [Image.fromarray(frame).convert(
                'RGB') for frame in video_data]
        except Exception:
            raise RuntimeError(f'Error occurred in reading frames {frame_id_list} from video {video_name} of duration {duration}. FFmpeg: {str(err)}')
        return sampled_list


def build_pretraining_dataset(args, load_ratio=0.8, spatial_refer=False):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        lazy_init=False,
        load_ratio=load_ratio,
        spatial_refer=spatial_refer)
    # effect of masked autoencoder directly depends on size of training dataset
    print("Data Aug = %s" % str(transform))
    return dataset
