import torch
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                          MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn,
                                          maskrcnn_resnet50_fpn_v2)

from mesh.dataset.kinetics import KINETICS400_DIR
from mesh.dnn_model.util import (convert_a_video_to_images_and_tensor,
                                 miou_accuracy, scale_segmentations, scale_segmentation_by_shape,
                                 show_segmentations, convert_tensor_to_an_image)

mask_rcnn_running = None
mask_rcnn_weights = None


def load_mask_rcnn(index=2, rank=0):
    score_thresh = 0.5
    if index == 1:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(
            weights=weights, box_score_thresh=score_thresh)
    elif index == 2:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model

def load_mask_rcnn_weights(index=2):
    global mask_rcnn_weights
    if mask_rcnn_weights is None:
        if index == 1:
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        elif index == 2:
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        mask_rcnn_weights = weights

def run_mask_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False, rank=0):
    global mask_rcnn_running, mask_rcnn_weights
    if mask_rcnn_running is None:
        mask_rcnn_weights, mask_rcnn_running = load_mask_rcnn(rank=rank)

    with torch.no_grad():
        results = mask_rcnn_running(inputs)
        if online is False:
            gt_results = mask_rcnn_running(gt_inputs)

    if show:
        show_segmentations(imgs, results, 0)
        show_segmentations(gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_segmentations(results, inputs, gt_inputs)
        acc = miou_accuracy(results, gt_results)
        print(acc)
        return acc
    else:
        return results

def inference_mask_rcnn(inputs, rank=0):
    global mask_rcnn_running, mask_rcnn_weights
    if mask_rcnn_running is None:
        mask_rcnn_weights, mask_rcnn_running = load_mask_rcnn(rank=rank)
    with torch.no_grad():
        results = mask_rcnn_running(inputs)
    return results

def visualize_mask_rcnn(tensor_frame, result, save_path):
    load_mask_rcnn_weights()
    img = convert_tensor_to_an_image(tensor_frame, False)
    show_segmentations([img], [result], 0, 0.5, save_path)

def accuracy_mask_rcnn(results, gt_results, input_shape, gt_shape):
    results = scale_segmentation_by_shape(results, input_shape, gt_shape)
    acc = miou_accuracy(results, gt_results)
    return acc

if __name__ == '__main__':
    imgs, inputs = convert_a_video_to_images_and_tensor(
        KINETICS400_DIR+'/k400_320p/Zz_VVFF9xMI_000001_000011.mp4')
    run_mask_rcnn(imgs, inputs, imgs, inputs)
