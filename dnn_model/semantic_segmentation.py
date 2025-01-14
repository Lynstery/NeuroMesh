import torch
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             DeepLabV3_ResNet101_Weights,
                                             FCN_ResNet50_Weights,
                                             FCN_ResNet101_Weights,
                                             deeplabv3_resnet50,
                                             deeplabv3_resnet101, fcn_resnet50,
                                             fcn_resnet101)

from mesh.dataset.kinetics import KINETICS400_DIR
from mesh.dnn_model.util import (convert_a_video_to_images_and_tensor,
                                 pixel_accuracy, scale_masks, scale_masks_by_shape, show_masks)

fcn_running = None
fcn_weights = None
deeplabv3_running = None
deeplabv3_weights = None


def load_fcn(index=2, rank=0):
    if index == 1:
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights)
    elif index == 2:
        weights = FCN_ResNet101_Weights.DEFAULT
        model = fcn_resnet101(weights=weights)
    model = model.to(rank)
    model.eval()
    return weights, model

def load_fcn_weights(index=2):
    global fcn_weights
    if fcn_weights is None:
        if index == 1:
            weights = FCN_ResNet50_Weights.DEFAULT
        elif index == 2:
            weights = FCN_ResNet101_Weights.DEFAULT
        fcn_weights = weights

def run_fcn(inputs, gt_inputs, show=True, online=False, rank=0):
    global fcn_running, fcn_weights
    if fcn_running is None:
        fcn_weights, fcn_running = load_fcn(rank=rank)

    with torch.no_grad():
        results = fcn_running(inputs)
        if online is False:
            gt_results = fcn_running(gt_inputs)

    if show:
        show_masks(fcn_weights, results, 0)
        show_masks(fcn_weights, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_masks(results, inputs, gt_inputs)
        acc = pixel_accuracy(results, gt_results)
        print(acc)
        return acc
    else:
        return results

def inference_fcn(inputs, rank=0):
    global fcn_running, fcn_weights
    if fcn_running is None:
        fcn_weights, fcn_running = load_fcn(rank=rank)
    with torch.no_grad():
        results = fcn_running(inputs)
    return results

def visualize_fcn(tensor_frame, result, save_path):
    load_fcn_weights()
    show_masks(fcn_weights, [result], 0, save_path)    

def accuracy_fcn(results, gt_results, input_shape, gt_shape):
    results = scale_masks_by_shape(results, input_shape, gt_shape)
    acc = pixel_accuracy(results, gt_results)
    return acc

def load_deeplabv3(index=2, rank=0):
    if index == 1:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    elif index == 2:
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights)
    model = model.to(rank)
    model.eval()
    return weights, model

def load_deeplabv3_weights(index=2):
    global deeplabv3_weights
    if deeplabv3_weights is None:
        if index == 1:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
        elif index == 2:
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
        deeplabv3_weights = weights

def run_deeplabv3(inputs, gt_inputs, show=True, online=False, rank=0):
    global deeplabv3_running, deeplabv3_weights
    if deeplabv3_running is None:
        deeplabv3_weights, deeplabv3_running = load_deeplabv3(rank=rank)

    with torch.no_grad():
        results = deeplabv3_running(inputs)
        if online is False:
            gt_results = deeplabv3_running(gt_inputs)

    if show:
        show_masks(deeplabv3_weights, results, 0)
        show_masks(deeplabv3_weights, gt_results, 0)

    if online is False:
        # print(results['out'])
        results = scale_masks(results, inputs, gt_inputs)
        acc = pixel_accuracy(results, gt_results)
        print(acc)
        return acc
    else:
        return results

def inference_deeplabv3(inputs, rank=0):
    global deeplabv3_running, deeplabv3_weights
    if deeplabv3_running is None:
        deeplabv3_weights, deeplabv3_running = load_deeplabv3(rank=rank)
    with torch.no_grad():
        results = deeplabv3_running(inputs)
    return results

def visualize_deeplabv3(tensor_frame, result, save_path):
    load_deeplabv3_weights()
    show_masks(deeplabv3_weights, [result], 0, save_path)

def accuracy_deeplabv3(results, gt_results, input_shape, gt_shape):
    results = scale_masks_by_shape(results, input_shape, gt_shape)
    acc = pixel_accuracy(results, gt_results)
    return acc

if __name__ == '__main__':
    imgs, inputs = convert_a_video_to_images_and_tensor(
        KINETICS400_DIR+'/k400_320p/ZnEfDMM-QpE_000174_000184.mp4')
    run_fcn(inputs, inputs)
    run_deeplabv3(inputs, inputs)
