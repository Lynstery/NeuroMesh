import os

import torch
from torch.hub import _import_module
from torchvision import transforms

from crucio.autoencoder.util import home
from crucio.dnn_model.util import (f1_score, scale_boxes_of_results,
                                   show_detections)

yolov5_running = None
nms_module = None
check_module = None
labels = None


def load_yolov5(weights=home+'/.cache/torch/hub/checkpoints/yolov5m.pt'):
    # rm -rf ~/.cache/torch/hub/ultralytics_yolov5_master
    yolov5 = home+'/.cache/torch/hub/ultralytics_yolov5_master'
    if os.path.isdir(yolov5):
        model = torch.hub.load(yolov5, 'custom', weights,
                               source='local', autoshape=False)
    else:
        weights = os.path.splitext(os.path.basename(weights))[0]
        model = torch.hub.load('ultralytics/yolov5', weights,
                               trust_repo=True, pretrained=True, autoshape=False)
    model.eval()
    nms_module = _import_module(
        name='non_max_suppression', path=yolov5+'/utils/general.py')
    check_module = _import_module(
        name='check_img_size', path=yolov5+'/utils/general.py')
    labels = model.names
    return model, nms_module, check_module, labels


def resize_img_for_yolo(img_tensor):
    '''
    For YOLOv5, image size must be multiple of max stride 32
    '''
    img_size = check_module.check_img_size(img_tensor.shape[-2:])
    resize = transforms.Resize(img_size, antialias=True)
    return resize(img_tensor)


def test_yolov5(imgs, inputs, gt_imgs, gt_inputs, show=True):
    global yolov5_running, nms_module, check_module, labels
    if yolov5_running is None:
        yolov5_running, nms_module, check_module, labels = load_yolov5()
    with torch.no_grad():
        inputs = resize_img_for_yolo(inputs)
        results = yolov5_running(inputs)
        gt_inputs = resize_img_for_yolo(gt_inputs)
        gt_results = yolov5_running(gt_inputs)
        results = nms_module.non_max_suppression(results)
        gt_results = nms_module.non_max_suppression(gt_results)

    if show:
        show_detections(labels, imgs, results, 0, True)
        show_detections(labels, gt_imgs, gt_results, 0, True)

    print(results[0])
    results = scale_boxes_of_results(results, inputs, gt_inputs, True)
    print(f1_score(results, gt_results, True))
