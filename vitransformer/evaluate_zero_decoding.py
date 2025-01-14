import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import argparse
from mesh.dataset.kinetics import build_pretraining_dataset
from mesh.dataset.lumpi import build_lumpi_multiview_dataset
from mesh.dataset.utils import DATA_DISK_DIR, MODEL_SAVE_DIR, ReshapeVideo
from mesh.dnn_model.object_detection import load_faster_rcnn, load_retinanet, RetinanetPost
from mesh.msc.utils import seed_worker
from mesh.vitransformer.arguments import get_args
from mesh.vitransformer.finetune import Block
from mesh.vitransformer.pretrain import __all__
from mesh.dnn_model.util import (convert_a_video_to_images_and_tensor, convert_a_video_tensor_to_images_list, show_detections)
from mesh.msc.utils import (DDP_PARAMETER_CHECK, NativeScalerWithGradNormCount,
                            TensorboardLogger, auto_load_model,
                            check_world_size, close_distributed_mode,
                            cosine_scheduler, get_rank, get_world_size,
                            init_distributed_mode, is_main_process, save_model,
                            seed_worker)
from mesh.dataset.utils import timm_video_normalization, KINETICS400_DIR
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

WORLD_SIZE, VISIBLE_IDS = check_world_size()

ZERO_DECODING_DIR = os.path.join(MODEL_SAVE_DIR, 'zero_decoding')
os.makedirs(ZERO_DECODING_DIR, exist_ok=True)


def scaling_three_dimensional_tensor_shapes(tensor, B, N, C):
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(
        B, N, C), mode='trilinear', align_corners=False)
    return tensor.squeeze(0).squeeze(0)


class ZeroDecodingInference(nn.Module):
    def __init__(self, num_patches, encoder_embed_dim=768, decoder_embed_dim=512, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, init_values=0., qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.dim = nn.Linear(2 * encoder_embed_dim,
                             decoder_embed_dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(decoder_embed_dim)

    def forward(self, original_patches, reference_patches, refer_infer_results):
        combined_patches = torch.cat(
            (original_patches, reference_patches), dim=2)
        combined_patches = self.dim(combined_patches)
        old_B, old_N, old_C = refer_infer_results.shape
        B, M, C = combined_patches.shape

        refer_infer_results = scaling_three_dimensional_tensor_shapes(
            refer_infer_results, B, self.num_patches - M, C)
        combined_patches = torch.cat(
            (combined_patches, refer_infer_results), dim=1)
        for blk in self.blocks:
            combined_patches = blk(combined_patches)
        pred_infer_results = self.norm(combined_patches)
        pred_infer_results = scaling_three_dimensional_tensor_shapes(
            pred_infer_results, old_B, old_N, old_C)

        return pred_infer_results


def load_inference_task(dnn_model, rank):
    if dnn_model == 2:
        return load_faster_rcnn(index=1, rank=rank)
    elif dnn_model == 3:
        return load_retinanet(index=1, rank=rank)
    else:
        raise ValueError("Unsupported DNN model")

def convert_results_to_tensor(results):
    cls_log, bbox_reg = results["cls_logits"], results["bbox_regression"]
    return torch.cat((cls_log, bbox_reg), dim=2)

def convert_results_to_dict(results):
    cls_log, bbox_reg = torch.split(results, [91, 4], dim=2)
    return {'cls_logits': cls_log, 'bbox_regression': bbox_reg}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args()

    rank = 2
    model = create_model(
        args.model,
        pretrained=True,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
        init_ckpt=os.path.join(
            DATA_DISK_DIR, 'mesh_model/1_vision/checkpoint-99.pth')
    )
    patch_size = model.encoder.patch_embed.patch_size

    model.eval()
    encoder = model.encoder.to(rank)
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    decoder = ZeroDecodingInference(num_patches=encoder.patch_embed.num_patches).to(rank)
    print("num_pathches:", encoder.patch_embed.num_patches) 
    state_dict = torch.load(os.path.join(ZERO_DECODING_DIR, 'decoder_epoch_99.pth'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # 去掉 module. 前缀
        new_state_dict[new_key] = v

    decoder.load_state_dict(new_state_dict)


    dataset = build_lumpi_multiview_dataset(args, load_ratio=1)
    inputs, ref_inputs = dataset[103] # normed
    inputs, ref_inputs = inputs.unsqueeze(0), ref_inputs.unsqueeze(0)

    #video_path1 = "/data/zh/LUMPI-dataset/Measurement5/cam/5/clips/5_0025.mp4"
    #video_path2 = "/data/zh/LUMPI-dataset/Measurement5/cam/6/clips/6_0025.mp4"
    #_, inputs = convert_a_video_to_images_and_tensor(video_path1, frame_num=16, video_tensor=True, new_size=(224,224))
    #_, ref_inputs = convert_a_video_to_images_and_tensor(video_path2, frame_num=16, video_tensor=True, new_size=(224,224))

    # 1, 3, 16, 224, 224 (norm)
    inputs = inputs.to(rank) 
    ref_inputs = ref_inputs.to(rank)

    if True:
        # unnorm 
        inputs_unnorm = timm_video_normalization(inputs, True)
        ref_inputs_unnorm = timm_video_normalization(ref_inputs, True)
        inputs_unnorm, ref_inputs_unnorm = inputs_unnorm.to(rank), ref_inputs_unnorm.to(rank)
    else:
        inputs_unnorm = inputs 
        ref_inputs_unnorm = ref_inputs

    # List[ PIL image ] (unnorm)
    imgs = convert_a_video_tensor_to_images_list(inputs_unnorm.squeeze(0)) 
    ref_imgs = convert_a_video_tensor_to_images_list(ref_inputs_unnorm.squeeze(0))

    # 16, 3, 224, 224 (unnorm)
    rev = ReshapeVideo(args.num_frames)
    inputs_frames, ref_inputs_frames = rev(inputs_unnorm), rev(ref_inputs_unnorm) 

    retinanet_weights, retinanet_running = load_inference_task(dnn_model=3, rank=rank)

    # encoder
    with torch.no_grad():
        original_patches, _, _ = encoder(inputs)
        reference_patches, _, _ = encoder(ref_inputs)

    # retinanet object detection without postprocess
    with torch.no_grad():
        original_results, _, _, _ = retinanet_running(inputs_frames, disable_post=True)
        refer_results, _, _, _ = retinanet_running(ref_inputs_frames, disable_post=True)

    original_results_tensor = convert_results_to_tensor(original_results)
    refer_results_tensor = convert_results_to_tensor(refer_results)

    # decoder predicts 
    with torch.no_grad():
        pred_results_tensor = decoder(original_patches, reference_patches, refer_results_tensor)

    # run postprocess to get labels
    retinanet_post = RetinanetPost(retinanet_running, inputs_frames)

    original_labels = retinanet_post.process(retinanet_running, original_results)
    refer_labels = retinanet_post.process(retinanet_running, refer_results)

    pred_results = convert_results_to_dict(pred_results_tensor)
    pred_labels = retinanet_post.process(retinanet_running, pred_results)

    show_detections(retinanet_weights, imgs, original_labels, 15, save='/data/zh/test/gt.png')
    show_detections(retinanet_weights, ref_imgs, refer_labels, 15, save='/data/zh/test/ref.png')
    show_detections(retinanet_weights, imgs, pred_labels, 15, save='/data/zh/test/pred.png')

    #with torch.no_grad():
    #    labels = retinanet_running(inputs_frames, disable_post=False)
    #show_detections(retinanet_weights, imgs, labels, 0, save='/data/zh/test/gt_straightforward.png')



    