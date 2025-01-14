import torch
import threading
import numpy as np
import ffmpeg
import cv2
import PIL.Image as Image
import io
import os
import tempfile
import time
import shutil
import lzma
import pickle
import subprocess
import random
from torchvision import transforms
from mesh.dataset.utils import DATA_DISK_DIR, timm_video_normalization, ReshapeVideo
from timm.models import create_model
from mesh.vitransformer.pretrain import __all__ 
from mesh.msc.engine_for_pretraining import generate_patchs_from_videos
from mesh.msc.engine_for_pretraining import (generate_patchs_from_videos, 
                                             reconstruct_videos_from_patchs_with_mean_std,
                                             video_frame_filtering)

from crucio.batch_handler.gru_filter import (get_filter, scores_to_selects)
from crucio.autoencoder.network3d import get_networks3d, CNN_FRAME_NUM
from mesh.baselines.TileClipper.src.tileClipper import TileClipper

class Codec:
    '''
        base class for video codec
        Tensor[1,3,num_frames,H,W] -> encode -> decode -> Tensor[1,3,num_rest_frames,H,W], select
    '''
    def __init__(self):
        pass
    
    def load_model(self, device):
        # need implementation in subclass
        raise NotImplementedError
    
    def encode_with_filter(self, video):
        # need implementation in subclass
        raise NotImplementedError
    
    def encode(self, video):
        # need implementation in subclass
        raise NotImplementedError
    
    def decode_with_filter(self, encoded_video):
        # need implementation in subclass
        raise NotImplementedError

    def decode(self, encoded_video):
        # need implementation in subclass
        raise NotImplementedError

def generate_binary_list(N, ratio):
    num_ones = int(N * ratio)
    num_zeros = N - num_ones
    # Create a list with the required number of 1s and 0s
    binary_list = [1] * num_ones + [0] * num_zeros
    random.shuffle(binary_list)  # Shuffle the list to randomize the order
    return np.array(binary_list, dtype=int)
    
class CrucioCodec(Codec):
    '''
        crucio codec
        crucio filter happens in the encoder side, so decode_with_filter is the same as decode
        Tensor[1,3,num_frames,H,W] -> encoder(with filter) -> [enc_output, select] -> decoder -> Tensor[1,3,rest_nums,H,W], select
        Tensor[1,3,num_frames,H,W] -> encoder(without filter) -> [enc_output, select=[1,1,..,1] ] -> decoder -> Tensor[1,3,num_frames,H,W], select=[1,..,1]
    '''
    def __init__(self):
        super().__init__()
        self.num_frames = 16
        self.is_model_loaded = False
        self.mu_load = threading.Lock()
        
    def load_model(self, device):
        with self.mu_load:
            if not self.is_model_loaded:
                self.device = device
                self.is_model_loaded = True
                self.encoder3d, _, self.decoder3d, _ = get_networks3d('eval', is_load=True, rank=self.device)
                self.extractor, self.gru = get_filter('eval', is_load=True, rank=self.device)

    def warmup(self):
        for _ in range(0, 5):
            inputs = torch.rand((1, 3, 16, 224, 224))
            encoded_video = self.encode_with_filter(inputs)
            _ = self.decode_with_filter(encoded_video)
        
    def decode(self, encoded_video):
        enc_outputs_list, select = encoded_video
        with torch.no_grad():
            decoded_video_list = []
            for enc_outputs in enc_outputs_list:
                enc_outputs = enc_outputs.to(self.device)
                decoded_video = self.decoder3d(enc_outputs)
                decoded_video_list.append(decoded_video)
        cat_decoded_video = torch.cat(decoded_video_list, dim=2) 
        rest_nums = np.sum(select)
        decoded_video = cat_decoded_video[:, :, :rest_nums, :, :]
        return decoded_video, select 
    
    def decode_with_filter(self, encoded_video):
        return self.decode(encoded_video)
    
    def encode(self, inputs):
        inputs = inputs.to(self.device)
        select = np.array([1]*self.num_frames)
        filtered_inputs = inputs.to(self.device)
        rest_nums = filtered_inputs.shape[2]
        split_list = [CNN_FRAME_NUM for _ in range(rest_nums // CNN_FRAME_NUM)]
        if rest_nums % CNN_FRAME_NUM > 0: 
            split_list.append(rest_nums % CNN_FRAME_NUM)
        split_inputs_list = list(filtered_inputs.split(split_list, dim=2))
        last_inputs = split_inputs_list[len(split_inputs_list)-1] 
        for _ in range(CNN_FRAME_NUM - (rest_nums % CNN_FRAME_NUM)):
            last_inputs = torch.cat([last_inputs, last_inputs[:, :, -1:, :, :]], dim=2)
        split_inputs_list[len(split_inputs_list)-1] = last_inputs
        enc_outputs_list = [] 
        with torch.no_grad():
            for inputs in split_inputs_list:
                enc_outputs = self.encoder3d(inputs)
                enc_outputs_list.append(enc_outputs)
        
        return [enc_outputs_list, select]
    
    def encode_with_filter(self, inputs):
        # inputs: 1, 3, num_frames, H, W 
        inputs = inputs.to(self.device)
        with torch.no_grad():
            features = self.extractor(inputs)
            scores = self.gru(features)

        selects = scores_to_selects(scores).detach().cpu().numpy()
        select = selects[0]
        select = np.array([int(x) for x in select])
        
        #select = random.select() 
        select = generate_binary_list(16, 0.8) 
       
        rev = ReshapeVideo(self.num_frames)   
        tensor_frames = rev(inputs)  # num_frames, 3, H, W
        filtered_frames = tensor_frames[select == 1] # rest_nums, 3, H, W
        filtered_inputs = filtered_frames.permute(1,0,2,3).unsqueeze(0) # 1, 3, rest_nums, H, W 
        
        rest_nums = filtered_inputs.shape[2]
        split_list = [CNN_FRAME_NUM for _ in range(rest_nums // CNN_FRAME_NUM)]
        if rest_nums % CNN_FRAME_NUM > 0: 
            split_list.append(rest_nums % CNN_FRAME_NUM)
        split_inputs_list = list(filtered_inputs.split(split_list, dim=2))
        last_inputs = split_inputs_list[len(split_inputs_list)-1] 
        for _ in range(CNN_FRAME_NUM - (rest_nums % CNN_FRAME_NUM)):
            last_inputs = torch.cat([last_inputs, last_inputs[:, :, -1:, :, :]], dim=2)
        split_inputs_list[len(split_inputs_list)-1] = last_inputs
        enc_outputs_list = [] 
        with torch.no_grad():
            for inputs in split_inputs_list:
                enc_outputs = self.encoder3d(inputs)
                enc_outputs_list.append(enc_outputs)
        return [enc_outputs_list, select]

def get_model_size(model):
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  # 参数个数 * 单个元素的大小（字节）
        param_count += param.numel()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2), param_count  # 返回模型大小（MB）和总参数个数


def calculate_inference_memory(model, input_size, dtype=torch.float32, batch_size=1):
    """
    计算模型推理所需显存大小
    Args:
        model (nn.Module): PyTorch 模型
        input_size (tuple): 输入张量的形状（不包括 batch size，例如 (3, 224, 224)）
        dtype (torch.dtype): 数据类型，例如 torch.float16, torch.float32
        batch_size (int): 批量大小
    Returns:
        dict: 包括参数显存、激活显存和总显存的估算结果（单位：MB）
    """
    device = 0 
    # 每个数据类型的字节大小
    element_size = torch.tensor([], dtype=dtype).element_size()
    
    # 1. 计算参数显存
    param_memory = sum(p.numel() for p in model.parameters()) * element_size

    # 2. 计算输入和输出张量显存
    input_memory = batch_size * torch.tensor(input_size).prod().item() * element_size
    dummy_input = torch.randn((batch_size, *input_size), dtype=dtype).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
        if isinstance(dummy_output, torch.Tensor):
            output_memory = dummy_output.numel() * element_size
        elif isinstance(dummy_output, (list, tuple)):
            output_memory = sum(o.numel() * element_size for o in dummy_output)
        else:
            output_memory = 0  # 自定义模型输出需要手动处理

    # 3. 激活显存估算（前向计算中间结果）
    activation_memory = 0
    hooks = []

    def activation_hook(module, input, output):
        nonlocal activation_memory
        if isinstance(output, torch.Tensor):
            activation_memory += output.numel() * element_size
        elif isinstance(output, (list, tuple)):
            activation_memory += sum(o.numel() * element_size for o in output)

    for layer in model.modules():
        hooks.append(layer.register_forward_hook(activation_hook))

    # 执行一次前向计算
    with torch.no_grad():
        model(dummy_input)

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    # 计算总显存需求
    total_memory = (param_memory + input_memory + output_memory + activation_memory) / (1024 ** 2)

    return {
        "Parameter Memory (MB)": param_memory / (1024 ** 2),
        "Input/Output Memory (MB)": (input_memory + output_memory) / (1024 ** 2),
        "Activation Memory (MB)": activation_memory / (1024 ** 2),
        "Total Memory (MB)": total_memory
    }

class MeshCodec(Codec):
    '''
        mesh codec
        mesh filter happens in the decoder side, so encode_with_filter is the same as encode
        Tensor[1,3,num_frames,H,W] -> encode -> [x_vis, mean, std, p_x, bool_masked_pos, visible_patchs] -> decoder(with filter) -> Tensor[1,3,num_rest,H,W], select
        Tensor[1,3,num_frames,H,W] -> encode -> [x_vis, mean, std, p_x, bool_masked_pos, visible_patchs] -> decoder(wo filter) -> Tensor[1,3,num_frames,H,W], select=[1...1]
    '''
    def __init__(self):
        super().__init__()
        self.num_frames = 16 
        self.mu_load = threading.Lock()
        self.is_model_loaded = False
    
    def load_model(self, device):
        with self.mu_load:
            if not self.is_model_loaded:
                self.is_model_loaded = True
                self.device = device
                init_ckpt=DATA_DISK_DIR+'/mesh_model/checkpoint_k400.pth'
                checkpoint = torch.load(init_ckpt, map_location="cpu", weights_only=False)
                state_dict = checkpoint['model']
                
                mesh_encoder = create_model(
                    'mesh_encode_model',
                    pretrained=True,
                    state_dict=state_dict,
                    drop_path_rate=0.0,
                    drop_block_rate=None,
                    decoder_depth=4,
                    mask_ratio=0.70,
                ).to(self.device)
                mesh_encoder.eval()

                mesh_decoder = create_model(
                    'mesh_decode_model',
                    pretrained=True,
                    state_dict=state_dict,
                    drop_path_rate=0.0,
                    drop_block_rate=None,
                    decoder_depth=4,
                    mask_ratio=0.70,
                ).to(self.device)
                mesh_decoder.eval()
                
                self.mesh_encoder = mesh_encoder
                self.mesh_decoder = mesh_decoder

                torch.jit.optimized_execution(True) 

                self.mesh_encoder = torch.jit.script(self.mesh_encoder)
                self.mesh_decoder = torch.jit.script(self.mesh_decoder)
                
                '''
                torch.jit.save(self.mesh_encoder, '/data/zh/mesh_model/scripted_mesh_encoder.pt')
                torch.jit.save(self.mesh_decoder, '/data/zh/mesh_model/scripted_mesh_decoder.pt')
                torch.jit.optimized_execution(True) 
                self.mesh_encoder = torch.jit.load(f'/data/{username}/mesh_model/scripted_mesh_encoder.pt')
                self.mesh_decoder = torch.jit.load(f'/data/{username}/mesh_model/scripted_mesh_decoder.pt')
                self.mesh_encoder = self.mesh_encoder.to(device) 
                self.mesh_decoder = self.mesh_decoder.to(device) 
                '''

                self.patch_size = 16
                self.input_size = 224

                print(f"load finished!") 
                
    def warmup(self):
        for _ in range(0, 5):
            inputs = torch.rand((1, 3, 16, 224, 224),dtype=torch.float32)
            t1 = time.time()
            encoded_video = self.encode_with_filter(inputs)
            t2 = time.time()
            print(f"encode time: {t2 - t1:.3f} s") 
            t1 = time.time()
            rec_video, select = self.decode_with_filter(encoded_video)
            t2 = time.time()
            print(f"decode time: {t2 - t1:.3f} s") 
        print("warmup finish!")
        
    def decode(self, encoded_video, decoder_undergo_depth=4):
        with torch.no_grad():
            encoded_video = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in encoded_video]
            x_vis, mean, std, p_x, bool_masked_pos, original_visible_patchs = encoded_video

            multilayer_outputs, content_complexity = self.mesh_decoder(x_vis, bool_masked_pos)
            #print(content_complexity)
            outputs = multilayer_outputs[decoder_undergo_depth-1]

            if original_visible_patchs is not None:
                # use original visible patchs
                _, N = bool_masked_pos.shape
                masked_num = torch.sum(bool_masked_pos[0]).item()
                outputs[:, :(N-masked_num)] = original_visible_patchs

            rec_videos = reconstruct_videos_from_patchs_with_mean_std(
                outputs, bool_masked_pos, self.num_frames, self.input_size, self.patch_size, mean, std)
            
            # do not filter 
            select = np.array([1]*self.num_frames) 
        return rec_videos, select

    def decode_with_filter(self, encoded_video, decoder_undergo_depth=4):
        with torch.no_grad():
            encoded_video = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in encoded_video]
            x_vis, mean, std, p_x, bool_masked_pos, original_visible_patchs = encoded_video
            # decode
            multilayer_outputs, content_complexity = self.mesh_decoder(x_vis, bool_masked_pos)
            #print(content_complexity)
            outputs = multilayer_outputs[decoder_undergo_depth-1]

            if original_visible_patchs is not None:
                # use original visible patchs
                _, N = bool_masked_pos.shape
                masked_num = torch.sum(bool_masked_pos[0]).item()
                outputs[:, :(N-masked_num)] = original_visible_patchs

            rec_videos = reconstruct_videos_from_patchs_with_mean_std(
                outputs, bool_masked_pos, self.num_frames, self.input_size, self.patch_size, mean, std)
            rev = ReshapeVideo(self.num_frames)
            filtered_frames, select = video_frame_filtering(rev, rec_videos, p_x, self.patch_size)
            # filtered_frames: rest_num, 3, H, W
            filtered_video = filtered_frames.permute(1,0,2,3).unsqueeze(0) # 1, 3, rest_num, H, W
            select = select.detach().cpu().numpy()
        return filtered_video, select
      
    def encode(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            original_patchs, mean, std = generate_patchs_from_videos(inputs, self.patch_size, True) 
            norm_inputs = timm_video_normalization(inputs)

            #x_vis, p_x, bool_masked_pos = self.model.encoder_forward(norm_inputs)
            x_vis, p_x, bool_masked_pos, content_complexity = self.mesh_encoder(norm_inputs)
            #print(content_complexity)
            visible_patchs = original_patchs[~bool_masked_pos]
            #visible_patchs = None
        '''
        print(f"x_vis: shape={x_vis.shape} type={x_vis.dtype}")
        print(f"mean: shape={mean.shape} type={mean.dtype}")
        print(f"std: shape={std.shape} type={std.dtype}")
        print(f"p_x: shape={p_x.shape} type={p_x.dtype}")
        print(f"bool_masked_pos: shape={bool_masked_pos.shape} type={bool_masked_pos.dtype}")
        print(f"visible_patches: shape={visible_patchs.shape} type={visible_patchs.dtype}")
        '''
        return [x_vis, mean, std, p_x, bool_masked_pos, visible_patchs]

    def encode_with_filter(self, inputs):
        return self.encode(inputs)

class TileClipperCodec(Codec):
    def __init__(self):
        super().__init__()
        self.num_frames = 16 
        
    def load_model(self, device):
        pass
    
    def prepare(self, imgs_dir):
        name = os.path.basename(imgs_dir.rstrip('/')) 
        tiled_path = os.path.join(imgs_dir, 'tiled') 
        script_dir = os.path.dirname(os.path.abspath(__file__))
        percentile_array_filename = os.path.join(script_dir, f"../baselines/TileClipper/assets/F2s/f2s_{name}_cluster10.pkl")
        cluster_indices_file = os.path.join(script_dir, f"../baselines/TileClipper/assets/F2s/{name}_cluster_indices.pkl")
        self.tileclipper = TileClipper()
        self.tileclipper.prepare(tiled_path, cluster_indices_file, percentile_array_filename) 

    def encode(self, video):
        imgs = [transforms.ToPILImage()(img) for img in video.squeeze(0).permute(1, 0, 2, 3)]
        frames = [np.array(img) for img in imgs]
        frames = np.stack(frames)  # Shape: (num_frames, height, width, channels)
        ffmpeg_input = frames.tobytes() 

        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{frames.shape[2]}x{frames.shape[1]}")
                .output('pipe:', pix_fmt='yuv420p', format='rawvideo')
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            kvazaar_process = subprocess.Popen(
                [
                    "kvazaar",
                    "-i", "-",  # Input from stdin
                    "--input-res", "224x224",
                    "--input-fps", "32",
                    "--qp", "30",
                    "--tiles", "4x4",
                    "--slice", "tiles",
                    "--mv-constraint", "frametilemargin",
                    "-o", f"{temp_dir}/seg_tiled.hevc"
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            ) 
            ffmpeg_output, ffmpeg_error = ffmpeg_process.communicate(input=ffmpeg_input)
            kvazaar_process.communicate(input=ffmpeg_output) 

            subprocess.run(f"MP4Box -add {temp_dir}/seg_tiled.hevc:split_tiles -new {temp_dir}/seg_tiled.mp4".split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.tileclipper.run(f"{temp_dir}/seg_tiled.mp4", f"{temp_dir}/result.mp4")

            with open(f"{temp_dir}/result.mp4", 'rb') as file:
                encoded_video = file.read()

        return encoded_video 
        
    def decode(self, encoded_video):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/result.mp4", 'wb') as file:
                file.write(encoded_video) 
            subprocess.run(["gpac", "-i", f'{temp_dir}/result.mp4', "tileagg", "@", "-o", f'{temp_dir}/result_agg.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
            subprocess.run(["ffmpeg", "-i", f'{temp_dir}/result_agg.mp4', '-v', 'error', f"{temp_dir}/%04d.png"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
            img_tensors = []
            for i in range(0, self.num_frames):
                img = Image.open(f"{temp_dir}/{i+1:04d}.png") 
                img_tensor = transforms.ToTensor()(img)
                img_tensors.append(img_tensor)
        video = torch.stack(img_tensors, dim=1).unsqueeze(0) # 1, 3, num_frames, 224, 224
        select = np.array([1]*self.num_frames) 
        return video, select
    
    def encode_with_filter(self, video):
        return self.encode(video)  

    def decode_with_filter(self, encoded_video):
        return self.decode(encoded_video)

class TraditionalCodec:
    def __init__(self):
        super().__init__()
        self.num_frames = 16 
    
    def load_model(self, device):
        pass
    
    def encode(self, video):
        imgs = [transforms.ToPILImage()(img) for img in video.squeeze(0).permute(1, 0, 2, 3)]
        frames = [np.array(img) for img in imgs]
        frames = np.stack(frames)
        ffmpeg_input = frames.tobytes()
        ffmpeg_process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{frames.shape[2]}x{frames.shape[1]}")
            .output('pipe:', vcodec='h264', pix_fmt='yuv420p', format='mp4', movflags='+frag_keyframe+empty_moov')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        encoded_video, _  = ffmpeg_process.communicate(input=ffmpeg_input) 
            
        return encoded_video

    def decode(self, encoded_video):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/output.mp4", 'wb') as file:
                file.write(encoded_video) 
            subprocess.run(f"ffmpeg -i {temp_dir}/output.mp4 {temp_dir}/%06d.png".split(' '),stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
            img_tensors = [] 
            for i in range(0, self.num_frames):
                img = Image.open(f"{temp_dir}/{i+1:06d}.png") 
                img_tensor = transforms.ToTensor()(img)
                img_tensors.append(img_tensor)
        video = torch.stack(img_tensors, dim=1).unsqueeze(0) # 1, 3, num_frames, 224, 224

        select = np.array([1]*self.num_frames) 
        return video, select 
    
    def decode_with_filter(self, encoded_video):
        return self.decode(encoded_video)

    def encode_with_filter(self, video):
        return self.encode(video)

username = os.getlogin()

def test_codec(codec_name, video, device=3):
    if codec_name == 'traditional':
        codec = TraditionalCodec()
        print("traditional codec")
    elif codec_name == 'tileclipper':
        codec = TileClipperCodec()
        codec.prepare(f'/data/{username}/videos_dir/AIC20-c001')
        print("TileClipper codec")
    elif codec_name == 'mesh':
        codec = MeshCodec()
        codec.load_model(device)
        codec.warmup()
        print("Mesh codec")
    elif codec_name == 'crucio':
        codec = CrucioCodec()
        codec.load_model(device)
        codec.warmup()
        print("Crucio codec")

    t1 = time.time()
    encoded_video = codec.encode_with_filter(video)
    t2 = time.time()
    print(f"encode time: {t2 - t1:.3f} s") 
    compressed_data = lzma.compress(pickle.dumps(encoded_video))
    print(f"data size: {len(compressed_data)/1024/1024} MB")
    if codec_name == 'mesh':
        compressed_data = lzma.compress(pickle.dumps(encoded_video[0]))
        print(f"data size (only x_vis): {len(compressed_data)/1024/1024} MB")
        
    t1 = time.time()
    rec_video, select = codec.decode_with_filter(encoded_video)
    #print(rec_video.shape)
    #print(select)
    t2 = time.time()
    print(f"decode time: {t2 - t1:.3f} s") 
    print("")

    os.makedirs(f"/data/{username}/test_codec", exist_ok=True)
    os.makedirs(f"/data/{username}/test_codec/{codec_name}", exist_ok=True)
    frames = rec_video.squeeze(0).permute(1,0,2,3) # num_frames, 3, H, W
    for i in range(frames.shape[0]):
        img = transforms.ToPILImage()(frames[i])
        img.save(f"/data/{username}/test_codec/{codec_name}/{i}.png")

if __name__=='__main__':
    img_tensors = []
    for i in range(100, 116):
        path = f'/data/{username}/videos_dir/AIC20-c001/{i:06d}.png'
        img = Image.open(path)
        img_tensor = transforms.Resize([224, 224], antialias=True)(transforms.ToTensor()(img)) 
        img_tensors.append(img_tensor)

    video = torch.stack(img_tensors, dim=1).unsqueeze(0) # 1, 3, 16, 224, 224

    test_codec(codec_name='mesh', video=video, device=0)
    test_codec(codec_name='crucio', video=video, device=0)
    #test_codec(codec_name='traditional', video=video, device=0)
    #test_codec(codec_name='tileclipper', video=video, device=0)

