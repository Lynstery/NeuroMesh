import math
import sys
from typing import Iterable

import torch
from einops import rearrange
from torch.autograd.function import InplaceFunction

from mesh.dataset.utils import ReshapeVideo, timm_video_normalization
from mesh.dnn_model.util import tensor_normalization
from mesh.msc.utils import MetricLogger, SmoothedValue

ENABLE_FILTERING = False
ENABLE_MULTILAYER_OUTPUT = False


def generate_patchs_from_videos(videos, patch_size, normlize_target=False, p0=2):
    if normlize_target is True:
        videos_squeeze = rearrange(
            videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=p0, p1=patch_size, p2=patch_size)
        mean = videos_squeeze.mean(dim=-2, keepdim=True)
        std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
        videos_norm = (videos_squeeze - mean) / (std + 1e-6)
        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
    else:
        mean = std = None
        videos_patch = rearrange(
            videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=p0, p1=patch_size, p2=patch_size)
    return videos_patch, mean, std

def get_video_patch_mean_std(videos, patch_size, p0=2):
    videos_squeeze = rearrange(
        videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=p0, p1=patch_size, p2=patch_size)
    mean = videos_squeeze.mean(dim=-2, keepdim=True)
    std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
    return mean, std    

def reconstruct_videos_from_patchs_with_mean_std(patchs, bool_masked_pos, num_frames, input_size, patch_size, mean, std, p0=2):
    B, input_N, C = patchs.shape
    B, N = bool_masked_pos.shape
    assert input_N == N
    masked_num = torch.sum(bool_masked_pos[0]).item()
    masked_patchs = patchs[:, -masked_num:]
    visible_patchs = patchs[:, :(N-masked_num)]
    videos_patch = torch.zeros((B, N, C), device=patchs.device)
    videos_patch[bool_masked_pos] = masked_patchs.reshape(-1, C).to(videos_patch.dtype)
    videos_patch[~bool_masked_pos] = visible_patchs.reshape(-1, C).to(videos_patch.dtype)
    videos_norm = rearrange(videos_patch, 'b n (p c) -> b n p c', c=3)
    videos_squeeze = videos_norm * std + mean
    videos = rearrange(videos_squeeze, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                        p0=p0, p1=patch_size, p2=patch_size, h=input_size//patch_size, w=input_size//patch_size)
    return videos.clamp(0, 1)


def reconstruct_videos_from_patchs(patchs, bool_masked_pos, num_frames, input_size, patch_size,
                                   original_videos, use_target_patch, normlize_target=False, p0=2):
    B, input_N, C = patchs.shape
    B, N = bool_masked_pos.shape
    masked_num = torch.sum(bool_masked_pos[0]).item()
    masked_patchs = patchs[:, -masked_num:]
    if original_videos is not None:
        original_patchs, mean, std = generate_patchs_from_videos(
            original_videos, patch_size, normlize_target)
    if input_N < N:
        visible_patchs = original_patchs[~bool_masked_pos]
    else:
        if use_target_patch:
            visible_patchs = original_patchs[~bool_masked_pos]
        else:
            visible_patchs = patchs[:, :(N-masked_num)]
    videos_patch = torch.zeros((B, N, C), device=patchs.device)
    videos_patch[bool_masked_pos] = masked_patchs.reshape(
        -1, C).to(videos_patch.dtype)
    videos_patch[~bool_masked_pos] = visible_patchs.reshape(
        -1, C).to(videos_patch.dtype)
    if normlize_target:
        videos_norm = rearrange(videos_patch, 'b n (p c) -> b n p c', c=3)
        videos_squeeze = videos_norm * std + mean
        videos = rearrange(videos_squeeze, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                           p0=p0, p1=patch_size, p2=patch_size, h=input_size//patch_size, w=input_size//patch_size)
    else:
        videos = rearrange(videos_patch, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=num_frames //
                           p0, h=input_size//patch_size, w=input_size//patch_size, p0=p0, p1=patch_size, p2=patch_size)
    return videos.clamp(0, 1)


class dff_round(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        result = torch.round(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return result * grad_output


def video_frame_filtering(rev, videos, p_x, patch_size, eval=True, p0=2, debug=False):
    # process output of decoder to obtain remaining frames after filtering
    B, C, T, H, W = videos.shape
    p_x = p_x.reshape(B, T//p0, H//patch_size, W//patch_size)
    p_x_norm = tensor_normalization(p_x.mean(dim=[2, 3]))
    if eval is True:
        assert B == 1
        p_x = p_x_norm[0]
        retain_index = 0
        # if (p_x < 0.5).any():
        #     condi = p_x < 0.5
        #     indices = torch.nonzero(condi)
        #     values = p_x[condi]
        #     distances = torch.abs(values - 0.5)
        #     retain_index = indices[torch.argmin(distances)]
        select = torch.round(p_x).repeat_interleave(p0, dim=0)
        # if debug:
        #     print(p_x, retain_index)
        #     assert p0*retain_index < select.shape[0]
        select[p0*retain_index] = 1  # frame with medium score are not filtered
        expanded_select = select.bool().unsqueeze(0).unsqueeze(
            1).unsqueeze(3).unsqueeze(4).expand(videos.shape)
        filtered_videos = videos[expanded_select].reshape(1, C, -1, H, W)
        video_frames = rev(filtered_videos)
        if debug:
            frame_index = 0
            for t in range(T):
                if select[t] == 1:
                    truth = videos[:, :, t:t+1, :, :].squeeze()
                    assert torch.sum(video_frames[frame_index]-truth) == 0
                    frame_index += 1
        return video_frames, select
    else:
        p_x_norm[:, 0] = 1.0  # the first frame cannot be filtered
        selects = dff_round.apply(p_x_norm).repeat_interleave(p0, dim=1)
        selects = selects.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        filtered_videos = videos * selects
        for b in range(B):
            reuse_frame = videos[b, :, 0:1, :, :]
            for t in range(T):
                current_frame = filtered_videos[b, :, t:t+1, :, :]
                if torch.sum(current_frame) == 0:
                    filtered_videos[b, :, t:t+1, :, :] = reuse_frame
                    # assert selects[b, :, t:t+1, :, :].item() == 0
                else:
                    reuse_frame = current_frame
                    # assert selects[b, :, t:t+1, :, :].item() == 1
                    # assert torch.sum(current_frame-videos[b, :, t:t+1, :, :]) == 0
        filter_ratio = (B*T-torch.sum(selects))/(B*T)
        return filtered_videos, filter_ratio


def train_one_epoch(loss_index, loss_func, rev, model: torch.nn.Module, data_loader: Iterable, acc_target,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    num_frames: int = 16, input_size: int = 224, patch_size: int = 16, normlize_target: bool = True,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * \
                        param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, _ = batch
        videos = videos.to(device, non_blocking=True)
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            unnorm_videos = timm_video_normalization(videos, True)
            videos_patch, _, _ = generate_patchs_from_videos(
                unnorm_videos, patch_size, normlize_target)

        with torch.amp.autocast("cuda"):
            multilayer_outputs, p_x, bool_masked_pos = model(videos)
            layer_num = len(multilayer_outputs)
            multilayer_loss = 0
            for L in range(layer_num-1, -1, -1):
                outputs = multilayer_outputs[L]
                rec_videos = reconstruct_videos_from_patchs(
                    outputs, bool_masked_pos, num_frames, input_size, patch_size, None, False)
                if ENABLE_FILTERING:
                    rec_videos, filter_ratio = video_frame_filtering(
                        rev, rec_videos, p_x, patch_size, False)

                # losses
                # Reconstruction loss: l_r -> B, N_m (for all tokens)
                if loss_index == 1:
                    rec_patch, _, _ = generate_patchs_from_videos(
                        rec_videos, patch_size)
                    mask_l_r = torch.mean(
                        loss_func(rec_patch, videos_patch), dim=-1)
                else:
                    if loss_index <= 7:
                        output_frames = rev(rec_videos)
                        video_frames = rev(videos)
                        mask_l_r, acc = loss_func(output_frames, video_frames)
                    else:
                        mask_l_r, acc = loss_func(rec_videos, videos)

                # Sampling loss: l_s -> B, N_m
                l_s = torch.zeros(videos.shape[0], ).to(mask_l_r.device)
                for i in range(p_x.shape[0]):
                    # categorical distribution
                    m = torch.distributions.categorical.Categorical(
                        probs=p_x[i])

                    # log-probabilities
                    log_probs = m.log_prob(torch.arange(
                        0, p_x.shape[1], 1).to(p_x.device))  # 1, N_m

                    # mask log-probs
                    mask_log_probs = log_probs[bool_masked_pos[i]]

                    # we need to select tokens that maximize the reconstruction error, so (-) sign
                    if loss_index == 1:
                        mask_l_r_value = mask_l_r[i].detach()[
                            bool_masked_pos[i]]
                    else:
                        mask_l_r_value = mask_l_r.item()
                    l_s[i] = -torch.mean(mask_log_probs * mask_l_r_value)

                # Total loss
                m_l_r = torch.mean(mask_l_r)  # Reconstruction loss
                m_l_s = 1e-4 * torch.mean(l_s)  # Sampling loss
                if ENABLE_FILTERING:
                    m_l_f = 1e-2 * (acc_target-1) * \
                        filter_ratio  # Filtering loss
                    loss = m_l_r + m_l_s + m_l_f
                else:
                    loss = m_l_r + m_l_s
                multilayer_loss += loss

                if ENABLE_MULTILAYER_OUTPUT is False:
                    break

        loss_value = multilayer_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(multilayer_loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_reconstruction=m_l_r.item(), head="loss")
            log_writer.update(loss_sampling=m_l_s.item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    rev = ReshapeVideo(16)
    for i in range(1000):
        videos = torch.rand(1, 3, 16, 224, 224)
        n = (16//2)*(224//16)*(224//16)
        p_x = torch.rand(1, n)
        video_frame_filtering(rev, videos, p_x, 16, debug=True)
