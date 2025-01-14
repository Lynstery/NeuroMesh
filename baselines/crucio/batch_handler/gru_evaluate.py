import os

import matplotlib.pyplot as plt
import torch
import ffmpeg
from crucio.autoencoder.dataset import (GRU_FRAME_NUM, IMAGE_EXT, VIDEO_DIR, KINETICS400_DIR, ReshapeVideo, load_video_to_tensor, load_from_encoded_video_to_tensor)
from crucio.autoencoder.util import (YUV_ENABLED, get_folder_size,
                                     load_compressed_data,
                                     save_compressed_data, save_compressed_data_at,
                                     save_tensor_to_video, )
from crucio.autoencoder.gpu import print_device_info 
from crucio.batch_handler.gru_filter import (get_filter, print_gru_info,
                                             scores_to_selects)
from crucio.dnn_model.faster_rcnn import test_faster_rcnn, run_fast_rcnn_on_filtered_frames, scale_boxes_of_results

def show_filter_results(number, video_path, selects):
    '''
    Display video filtering (1 for keyframe and 0 for filtered frame)
    number -> Number of frames in video
    video_path -> Absolute path to video directory
    '''
    # Create a new image window
    fig = plt.figure(figsize=(12, 6))

    # Add subgraph for first row and set title
    for i in range(1, number+1):
        str_num = "{:04d}".format(i)
        img_path = os.path.join(video_path, 'frame'+str_num + IMAGE_EXT)
        ax = fig.add_subplot(1, number, i)
        ax.set_title('Frame'+str(i)+'='+str(int(selects[i-1])))
        ax.imshow(plt.imread(img_path))

    # Adjust spacing between subgraphs
    plt.subplots_adjust(wspace=0.05)

    # Show images
    plt.show()


print_device_info()
print_gru_info()


# load trained GRU encoder
extractor, gru = get_filter('eval', True)

def crucio_filter(video_tensor):
    # 1, 3, GRU_FRAME_NUM, H, W 
    with torch.no_grad():
        features = extractor(video_tensor)
        scores = gru(features)

    selects = scores_to_selects(scores).detach().cpu().numpy()
    return selects

if __name__ == '__main__':

    '''
    # GRU filter test video
    video_path = VIDEO_DIR+'/aeroplane_0001_039'
    video_tensor = load_video_to_tensor(
        video_path, length=GRU_FRAME_NUM).unsqueeze(0)
    '''
    video_path = KINETICS400_DIR+'/k400_320p/t_hQbWfSsP0_000084_000094.mp4'
    base_name = os.path.basename(video_path)
    base_name_without_ext = os.path.splitext(base_name)[0]

    probe = ffmpeg.probe(video_path)
    duration = int(probe['streams'][0]['nb_frames'])
    print("duration: ", duration)
    video_tensor = load_from_encoded_video_to_tensor(
        video_path=video_path, probe=probe, duration=duration, offset=1, num_frames=GRU_FRAME_NUM, step=10, random_step=True, is_yuv=False).unsqueeze(0)

    # Save original video
    origin_path = f"/data/zh/test/origin/{base_name_without_ext}"
    save_tensor_to_video(origin_path, video_tensor[0], False)
    print(f"Test video saved at {origin_path}")

    print(video_tensor.shape)
    selects = crucio_filter(video_tensor)
    selects = selects[0]
    print(selects)
    show_filter_results(GRU_FRAME_NUM, origin_path, selects)
    rev = ReshapeVideo()
    tensor_frames = rev(video_tensor).squeeze(0)  # GRU_FRAME_NUM, 3, H, W
    results = run_fast_rcnn_on_filtered_frames(tensor_frames, selects)
    print(results)

