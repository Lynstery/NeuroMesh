import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp

from mesh.dataset.kinetics import build_pretraining_dataset
from mesh.dataset.lumpi import build_lumpi_multiview_dataset
from mesh.dataset.utils import DATA_DISK_DIR, MODEL_SAVE_DIR, ReshapeVideo
from mesh.dnn_model.object_detection import load_faster_rcnn, load_retinanet
from mesh.msc.utils import seed_worker
from mesh.vitransformer.arguments import get_args
from mesh.vitransformer.finetune import Block
from mesh.vitransformer.pretrain import __all__
from mesh.dataset.utils import timm_video_normalization
from mesh.msc.utils import (DDP_PARAMETER_CHECK, NativeScalerWithGradNormCount,
                            TensorboardLogger, auto_load_model,
                            check_world_size, close_distributed_mode,
                            cosine_scheduler, get_rank, get_world_size,
                            init_distributed_mode, is_main_process, save_model,
                            seed_worker)
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


def process_batch(encoder, inference_task, rev, original_videos, reference_videos, rank):
    original_videos, reference_videos = original_videos.to(rank), reference_videos.to(rank)
    original_videos_unnorm = timm_video_normalization(original_videos, True)
    reference_videos_unnorm = timm_video_normalization(reference_videos, True)
    original_frames, reference_frames = rev(original_videos_unnorm), rev(reference_videos_unnorm)

    with torch.no_grad():
        original_patches, _, _ = encoder(original_videos)
        reference_patches, _, _ = encoder(reference_videos)
        original_infer_results, _, _, _ = inference_task(original_frames, disable_post=True)
        refer_infer_results, _, _, _ = inference_task(reference_frames, disable_post=True)

    # ? only support retinanet
    o_cls_log, o_bbox_reg = original_infer_results["cls_logits"], original_infer_results["bbox_regression"]
    original_infer_results = torch.cat((o_cls_log, o_bbox_reg), dim=2)
    r_cls_log, r_bbox_reg = refer_infer_results["cls_logits"], refer_infer_results["bbox_regression"]
    refer_infer_results = torch.cat((r_cls_log, r_bbox_reg), dim=2)

    return original_patches, reference_patches, refer_infer_results, original_infer_results


def train_variant_decoder(rank, dnn_model, encoder, decoder, train_loader, rev, optimizer, criterion, num_epochs, opts):
    _, inference_task = load_inference_task(dnn_model, rank)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    for epoch in range(num_epochs):
        if opts.distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0

        for batch_idx, (original_videos, reference_videos) in enumerate(train_loader):

            original_patches, reference_patches, refer_infer_results, original_infer_results = process_batch(encoder, inference_task, rev, original_videos, reference_videos, rank)
            pred_infer_results = decoder(original_patches, reference_patches, refer_infer_results)
            loss = criterion(pred_infer_results, original_infer_results)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            epoch_loss += loss.item()

            if rank == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            del original_patches, reference_patches, refer_infer_results, original_infer_results, loss
            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(train_loader)

        if rank == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}')

            if epoch % 5 == 4: 
                torch.save(decoder.state_dict(), os.path.join(ZERO_DECODING_DIR, f'decoder_epoch_{epoch}.pth'))
                print(f'Model weights saved for epoch {epoch + 1}')

        scheduler.step()

    close_distributed_mode(opts)

def main(rank, opts):
    init_distributed_mode(WORLD_SIZE, VISIBLE_IDS, rank, opts)

    # fix the seed for reproducibility
    seed = opts.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed) 

    model = create_model(
        opts.model,
        pretrained=True,
        drop_path_rate=opts.drop_path,
        drop_block_rate=None,
        decoder_depth=opts.decoder_depth,
        mask_ratio=opts.mask_ratio,
        init_ckpt=os.path.join(
            DATA_DISK_DIR, 'mesh_model/1_vision/checkpoint-99.pth')
    )
    patch_size = model.encoder.patch_embed.patch_size
    opts.patch_size = patch_size
    opts.window_size = (opts.num_frames // 2, opts.input_size //
                        patch_size[0], opts.input_size // patch_size[1])
    model.eval()
    encoder = model.encoder.to(rank)

    decoder = ZeroDecodingInference(
        num_patches=encoder.patch_embed.num_patches).to(rank)
    
    decoder_without_ddp = decoder

    #dataset_train = build_pretraining_dataset(opts, load_ratio=0.01, spatial_refer=True)
    dataset_train = build_lumpi_multiview_dataset(opts, load_ratio=1)

    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler_rank = global_rank
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        persistent_workers=True,
        pin_memory=opts.pin_mem,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    total_batch_size = opts.batch_size * get_world_size()

    print(f"total batch size: {total_batch_size}")
    device_id = torch.cuda.current_device()
    if opts.distributed:
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder, device_ids=[opts.gpu], find_unused_parameters=DDP_PARAMETER_CHECK)
        decoder_without_ddp = decoder.module
    rev = ReshapeVideo(opts.num_frames)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    dnn_model = 3
    torch.cuda.empty_cache()
    train_variant_decoder(rank, dnn_model, encoder, decoder,
                          train_loader, rev, optimizer, criterion, num_epochs=100, opts=opts)


if __name__ == "__main__":
    opts = get_args()
    opts.batch_size = 1
    opts.num_frames = 16

    if WORLD_SIZE == 1:
        main(VISIBLE_IDS[0], opts)
    else:
        try:
            mp.spawn(main, args=(opts,), nprocs=WORLD_SIZE)
        except Exception as e:
            print(e)
