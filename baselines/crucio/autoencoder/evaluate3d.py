import os
import ffmpeg
import torch

from crucio.autoencoder.dataset import (CNN_FRAME_NUM, VIDEO_DIR, KINETICS400_DIR,
                                        load_video_to_tensor, load_from_encoded_video_to_tensor,
                                        show_videos_difference)
from crucio.autoencoder.network3d import (get_networks3d,
                                          print_autoencoder3d_info)
from crucio.autoencoder.util import (YUV_ENABLED, get_folder_size,
                                     load_compressed_data,
                                     save_compressed_data, save_compressed_data_at,
                                     save_tensor_to_video)
from crucio.autoencoder.gpu import print_device_info

print_device_info()
print_autoencoder3d_info()

# Load trained encoder and decoder
encoder3d, _, decoder3d, _ = get_networks3d('eval', True)

def networks3d_encode(videos):
    # videos: 1, 3, CNN_FRAME_NUM, H, W 
    with torch.no_grad():
        compressed_data = encoder3d(videos)
    return compressed_data

def networks3d_decode(compressed_data):
    with torch.no_grad():
        decoded_tensor = decoder3d(compressed_data)
    return decoded_tensor

if __name__ == '__main__':

    video_path = KINETICS400_DIR+'/k400_320p/Ab526k1iFdc_000002_000012.mp4'
    base_name = os.path.basename(video_path)
    base_name_without_ext = os.path.splitext(base_name)[0]

    probe = ffmpeg.probe(video_path)
    duration = int(probe['streams'][0]['nb_frames'])
    video_tensor = load_from_encoded_video_to_tensor(
        video_path=video_path, probe=probe, duration=duration, offset=1, num_frames=5, step=5, is_yuv=YUV_ENABLED).unsqueeze(0)
    print(f"video tensor shape: {video_tensor.shape}")

    # Save original video
    origin_path = f"/data/zh/test/origin/{base_name_without_ext}"
    save_tensor_to_video(origin_path, video_tensor[0], False)
    print(f"Test video saved at {origin_path}")
    video_size = get_folder_size(origin_path)
    print(f"Size of original video {origin_path} is {video_size:.4f} KB")

    # Encoder compresses test video
    compressed_data = networks3d_encode(video_tensor)

    # Save compressed data
    data_path = save_compressed_data_at(f"/data/zh/test/compressed/{base_name_without_ext}.pkl", compressed_data)
    data_size = os.path.getsize(data_path)/1024
    print(f"Size of compressed data {data_path} is {data_size:.4f} KB")

    # Load compressed data
    compressed_data = load_compressed_data(data_path)

    # decode compressed data  
    decoded_tensor = networks3d_decode(compressed_data)

    print("Decoded tensor shape:", decoded_tensor.shape)

    # Save reconstructed video
    reconstructed_path = f"/data/zh/test/reconstructed/{base_name_without_ext}" 
    save_tensor_to_video(reconstructed_path, decoded_tensor[0], False)
    reconstructed_size = get_folder_size(reconstructed_path)
    print(f"Size of reconstructed video {reconstructed_path} is {reconstructed_size:.4f} KB")

    show_videos_difference(5, origin_path, reconstructed_path)