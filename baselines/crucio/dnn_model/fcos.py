import torch
from torchvision.models.detection.fcos import (FCOS_ResNet50_FPN_Weights,
                                               fcos_resnet50_fpn)

from crucio.dnn_model.util import (f1_score, scale_boxes_of_results,
                                   show_detections)

fcos_running = None
fcos_weights = None


def load_fcos(rank=0):
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights, score_thresh=0.5)
    model = model.to(rank)
    model.eval()
    return weights, model


def test_fcos(imgs, inputs, gt_imgs, gt_inputs, show=True):
    global fcos_running, fcos_weights
    if fcos_running is None:
        fcos_weights, fcos_running = load_fcos()
    with torch.no_grad():
        results = fcos_running(inputs)
        gt_results = fcos_running(gt_inputs)

    if show:
        show_detections(fcos_weights, imgs, results, 0)
        show_detections(fcos_weights, gt_imgs, gt_results, 0)

    print(results[0])
    results = scale_boxes_of_results(results, inputs, gt_inputs)
    print(f1_score(results, gt_results))
