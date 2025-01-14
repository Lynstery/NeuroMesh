import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io.image import read_image
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from crucio.autoencoder.dataset import IMAGE_DIR
from crucio.autoencoder.util import (CUDA_ENABLED, IMAGE_EXT,
                                     convert_image_colorspace,
                                     convert_image_to_tensor,
                                     convert_tensor_to_image)


def get_boxes_and_labels(results, index, is_yolov5):
    '''
    Return bounding boxes and labels for inference results
    results -> A batch of inference results
    index -> Detection index in a batch
    is_yolov5 -> Whether analytics task is YOLOv5
    '''
    if is_yolov5:
        boxes = results[index][:, 0:4]
        labels = results[index][:, 5]
    else:
        boxes = results[index]["boxes"]
        labels = results[index]["labels"]
    return boxes, labels


def show_detections(weights_or_labels, imgs, results, index, is_yolov5=False):
    '''
    Inference results are displayed as images with bounding boxes
    weights_or_labels -> Model weights or label names of analytics task
    imgs -> A batch of input images
    results -> A batch of inference results
    index -> Detection index in a batch
    '''
    boxes, labels = get_boxes_and_labels(results, index, is_yolov5)
    if is_yolov5:
        names = [weights_or_labels[int(i)] for i in labels.cpu()]
    else:
        names = [weights_or_labels.meta["categories"][i] for i in labels]
    boxes = draw_bounding_boxes(read_image(
        imgs[index]), boxes, names, width=4, font='timesbd.ttf', font_size=22)
    img = to_pil_image(boxes)
    img.show()


def scale_boxes_of_results(results, inputs, target, is_yolov5=False):
    '''
    Scale bounding boxes of inference results based on target images
    results -> A batch of inference results
    inputs -> A batch of input images
    target -> A batch of target images
    is_yolov5 -> Whether analytics task is YOLOv5
    '''
    image_shapes = inputs.shape[-2:]
    target_image_sizes = target.shape[-2:]
    if image_shapes != target_image_sizes:
        for i in range(len(results)):
            boxes, _ = get_boxes_and_labels(results, i, is_yolov5)
            boxes = resize_boxes(boxes, image_shapes, target_image_sizes)
            if is_yolov5:
                results[i][:, 0:4] = boxes
            else:
                results[i]["boxes"] = boxes
    return results


def f1_score(results, gt_results, is_yolov5=False, min_iou=0.7):
    '''
    Calculating F1 score [0,1] of current results based on Ground Truth 
    results -> A batch of inference results
    gt_results -> A batch of inference results (Ground Truth)
    is_yolov5 -> Whether analytics task is YOLOv5
    min_iou -> Minimum IOU requirements for bounding box overlap
    '''
    assert len(results) == len(gt_results)
    number = len(gt_results)
    f1 = []
    for i in range(number):
        boxes, labels = get_boxes_and_labels(results, i, is_yolov5)
        tp_and_fp = boxes.shape[0]
        gt_boxes, gt_labels = get_boxes_and_labels(gt_results, i, is_yolov5)
        tp_and_fn = gt_boxes.shape[0]
        score = 0
        if tp_and_fn == 0:
            score = 1
        else:
            if tp_and_fp != 0:
                iou = box_iou(boxes, gt_boxes)
                best_match = torch.nonzero(iou >= min_iou).tolist()
                tp = 0
                for j in range(len(best_match)):
                    index = best_match[j][0]
                    gt_index = best_match[j][1]
                    if labels[index] == gt_labels[gt_index]:
                        tp += 1
                # tp/(tp+fp)
                precision = tp/tp_and_fp
                # tp/(tp+fn)
                recall = tp/tp_and_fn
                if tp != 0:
                    score = 2/(1/precision+1/recall)
        f1.append(score)
    return np.mean(f1)


def map_accuracy(results, gt_results, is_yolov5=False):
    # TODO: Normalize confidences of results based on gt_results
    '''
    Calculating MAP [0,1] of current results based on Ground Truth
    results -> A batch of inference results
    gt_results -> A batch of inference results (Ground Truth)
    is_yolov5 -> Whether analytics task is YOLOv5
    '''
    metric = MeanAveragePrecision()
    if is_yolov5:
        comparison = []
        ground_truth = []
        batch_size = len(results)
        for _ in range(batch_size):
            boxes = results[_][:, 0:4]
            scores = results[_][:, 4]
            labels = results[_][:, 5].int()
            gt_boxes = gt_results[_][:, 0:4]
            gt_labels = gt_results[_][:, 5].int()
            comparison.append(
                {'boxes': boxes, 'scores': scores, 'labels': labels})
            ground_truth.append({'boxes': gt_boxes, 'labels': gt_labels})
    else:
        comparison = results
        ground_truth = [{k: v for k, v in d.items() if k != 'scores'}
                        for d in gt_results]
    metric.update(comparison, ground_truth)
    # global mean average precision
    accuracy = metric.compute()['map']
    # mean average precision at IoU=0.50
    # accuracy = metric.compute()['map_50']
    # mean average precision at IoU=0.75
    # accuracy = metric.compute()['map_75']
    return accuracy.item()


def test_convert_function(image_path, is_yuv, is_gpu):
    '''
    Test correctness of images and tensor conversion functions
    '''
    tensor = convert_image_to_tensor(image_path, is_yuv, is_gpu)
    tensor1 = convert_image_colorspace(tensor.unsqueeze(0), not is_yuv)[0]
    tensor2 = convert_image_colorspace(tensor1.unsqueeze(0), is_yuv)[0]
    comparison = torch.abs(tensor - tensor2)
    assert torch.all(comparison <= 1e-2)
    print(torch.sum(comparison))
    converted_img = convert_tensor_to_image(tensor1, not is_yuv)
    converted_img.save('converted_image'+IMAGE_EXT)
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(image_path))
    plt.title('Test Image')
    plt.subplot(1, 2, 2)
    plt.imshow(plt.imread('converted_image'+IMAGE_EXT))
    plt.title('Converted Image')
    plt.show()
    os.remove('converted_image'+IMAGE_EXT)


if __name__ == '__main__':
    test_convert_function(IMAGE_DIR+'/000000013291' +
                          IMAGE_EXT, True, CUDA_ENABLED)
    test_convert_function(IMAGE_DIR+'/000000013291' +
                          IMAGE_EXT, False, CUDA_ENABLED)
