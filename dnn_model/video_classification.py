import torch
from torchvision.models.video import (MC3_18_Weights, R2Plus1D_18_Weights,
                                      R3D_18_Weights, S3D_Weights, mc3_18,
                                      r2plus1d_18, r3d_18, s3d)

from mesh.dataset.kinetics import KINETICS400_DIR
from mesh.dnn_model.util import (category_accuracy,
                                 convert_a_video_to_images_and_tensor)

video_resnet_running = None
video_resnet_weights = None
video_s3d_running = None
video_s3d_weights = None


def load_video_resnet(index=2, rank=0):
    if index == 1:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    elif index == 2:
        weights = MC3_18_Weights.DEFAULT
        model = mc3_18(weights=weights)
    elif index == 3:
        weights = R2Plus1D_18_Weights.DEFAULT
        model = r2plus1d_18(weights=weights)
    model = model.to(rank)
    model.eval()
    return weights, model


def run_video_resnet(inputs, gt_inputs, show=True, online=False, rank=0):
    global video_resnet_running, video_resnet_weights
    if video_resnet_running is None:
        video_resnet_weights, video_resnet_running = load_video_resnet(
            rank=rank)

    with torch.no_grad():
        results = video_resnet_running(inputs)
        if online is False:
            gt_results = video_resnet_running(gt_inputs)

    if online is False:
        acc = category_accuracy(video_resnet_weights,
                                results, gt_results, show)
        print(acc)
        return acc
    else:
        return results

def inference_video_resnet(inputs, rank=0):
    global video_resnet_running, video_resnet_weights
    if video_resnet_running is None:
        video_resnet_weights, video_resnet_running = load_video_resnet(rank=rank)
    with torch.no_grad():
        results = video_resnet_running(inputs)
    return results

def visualize_video_resnet(tensor_frames, results, save_path):
    pass

def accuracy_video_resnet(results, gt_results):
    acc = category_accuracy(video_resnet_weights, results, gt_results, False)
    return acc


def load_video_s3d(rank=0):
    weights = S3D_Weights.DEFAULT
    model = s3d(weights=weights)
    model = model.to(rank)
    model.eval()
    return weights, model


def run_video_s3d(inputs, gt_inputs, show=True, online=False, rank=0):
    global video_s3d_running, video_s3d_weights
    if video_s3d_running is None:
        video_s3d_weights, video_s3d_running = load_video_s3d(rank=rank)

    assert gt_inputs.shape[2] >= 13
    # Video S3D模型要求输入视频的帧数量不低于13
    with torch.no_grad():
        results = video_s3d_running(inputs)
        if online is False:
            gt_results = video_s3d_running(gt_inputs)

    if online is False:
        acc = category_accuracy(video_s3d_weights, results, gt_results, show)
        print(acc)
        return acc
    else:
        return results

def inference_video_s3d(inputs, rank=0):
    global video_s3d_running, video_s3d_weights
    if video_s3d_running is None:
        video_s3d_weights, video_s3d_running = load_video_s3d(rank=rank)
    with torch.no_grad():
        results = video_s3d_running(inputs)
    return results

def visualize_video_s3d(tensor_frames, results, save_path):
    pass

def accuracy_video_s3d(results, gt_results):
    acc = category_accuracy(video_s3d_weights, results, gt_results, False)
    return acc


if __name__ == '__main__':
    imgs, inputs = convert_a_video_to_images_and_tensor(
        KINETICS400_DIR+'/k400_320p/ZNAnIDp2QJg_000004_000014.mp4', 13, video_tensor=True)
    for _ in range(13):
        imgs[_].show()
    run_video_resnet(inputs, inputs)
    run_video_s3d(inputs, inputs)
