import torch
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2)

from mesh.dataset.kinetics import KINETICS400_DIR
from mesh.dnn_model.util import (convert_a_video_to_images_and_tensor, f1_score,
                                 scale_boxes_of_results, scale_boxes_of_results_by_shape, show_detections, convert_tensor_to_an_image)
from mesh.dataset.utils import timm_video_normalization, ReshapeVideo

faster_rcnn_running = None
faster_rcnn_weights = None
retinanet_running = None
retinanet_weights = None


def load_faster_rcnn(index=2, rank=0):
    score_thresh = 0.5
    if index == 1:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(
            weights=weights, box_score_thresh=score_thresh)
    elif index == 2:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model

def load_faster_rcnn_weights(index=2):
    global faster_rcnn_weights
    if faster_rcnn_weights is None:
        if index == 1:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        elif index == 2:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        faster_rcnn_weights = weights        

def run_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False, rank=0):
    global faster_rcnn_running, faster_rcnn_weights
    if faster_rcnn_running is None:
        faster_rcnn_weights, faster_rcnn_running = load_faster_rcnn(rank=rank)

    with torch.no_grad():
        results = faster_rcnn_running(inputs)
        if online is False:
            gt_results = faster_rcnn_running(gt_inputs)

    if show:
        show_detections(faster_rcnn_weights, imgs, results, 0)
        show_detections(faster_rcnn_weights, gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_boxes_of_results(results, inputs, gt_inputs)
        acc = f1_score(results, gt_results)
        print(acc)
        return acc
    else:
        return results

def filter_classes(predictions):
    allowed_classes = [1, 3]
    filtered_predictions = [
        {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }
        for pred in predictions
        for boxes, scores, labels in [(pred["boxes"], pred["scores"], pred["labels"])]
        for keep in [torch.isin(labels, torch.tensor(allowed_classes).to(labels.device))]
    ]
    return filtered_predictions

def filter_small_targets(predictions, min_area=700):
    filtered_predictions = []
    for pred in predictions:
        areas = (pred['boxes'][:, 2] - pred['boxes'][:, 0]) * (pred['boxes'][:, 3] - pred['boxes'][:, 1])
        keep = areas > min_area
        filtered_predictions.append({
            "boxes": pred['boxes'][keep],
            "scores": pred['scores'][keep],
            "labels": pred['labels'][keep],
        })
    return filtered_predictions

def inference_faster_rcnn(inputs, rank=0):
    global faster_rcnn_running, faster_rcnn_weights
    if faster_rcnn_running is None:
        faster_rcnn_weights, faster_rcnn_running = load_faster_rcnn(rank=rank)

    with torch.no_grad():
        results = faster_rcnn_running(inputs)

    results = filter_classes(results)
    results = filter_small_targets(results)
    return results

def visualize_faster_rcnn(tensor_frame, result, save_path):
    load_faster_rcnn_weights()
    img = convert_tensor_to_an_image(tensor_frame, False)
    show_detections(faster_rcnn_weights, [img], [result], 0, False, save_path)

def accuracy_faster_rcnn(results, gt_results, input_shape, gt_shape):
    results = scale_boxes_of_results_by_shape(results, input_shape, gt_shape)
    acc = f1_score(results, gt_results)
    return acc


def load_retinanet(index=2, rank=0):
    score_thresh = 0.5
    if index == 1:
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = retinanet_resnet50_fpn(
            weights=weights, score_thresh=score_thresh)
    elif index == 2:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(
            weights=weights, score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model

def load_retinanet_weights(index=2):
    global retinanet_weights
    if retinanet_weights is None:
        if index == 1:
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        elif index == 2:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        retinanet_weights = weights

class RetinanetPost:

    original_image_sizes, num_anchors_per_level, anchors, image_sizes = None, None, None, None

    def __init__(self, retinanet_running, inputs_example):
        
        self.original_image_sizes = []

        for img in inputs_example:
            val = img.shape[-2:]
            self.original_image_sizes.append((val[0], val[1]))

        with torch.no_grad():
            head_outputs, self.num_anchors_per_level, self.anchors, self.image_sizes = retinanet_running(inputs_example, disable_post=True)

    def process(self, retinanet_running, head_outputs):
        results = retinanet_running.forward_post(head_outputs, self.num_anchors_per_level, self.anchors, self.image_sizes, self.original_image_sizes)
        return results


def run_retinanet(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False, rank=0):
    global retinanet_running, retinanet_weights
    if retinanet_running is None:
        retinanet_weights, retinanet_running = load_retinanet(rank=rank)

    with torch.no_grad():
        results = retinanet_running(inputs)
        if online is False:
            gt_results = retinanet_running(gt_inputs)

    if show:
        show_detections(retinanet_weights, imgs, results, 0)
        show_detections(retinanet_weights, gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_boxes_of_results(results, inputs, gt_inputs)
        acc = f1_score(results, gt_results)
        print(acc)
        return acc
    else:
        return results

def inference_retinanet(inputs, rank=0):
    global retinanet_running, retinanet_weights
    if retinanet_running is None:
        retinanet_weights, retinanet_running = load_retinanet(rank=rank)

    with torch.no_grad():
        results = retinanet_running(inputs, disable_post=False)

    return results

def visualize_retinanet(tensor_frame, result, save_path):
    load_retinanet_weights()
    img = convert_tensor_to_an_image(tensor_frame, False)
    show_detections(retinanet_weights, [img], [result], 0, False, save_path)

def accuracy_retinanet(results, gt_results, input_shape, gt_shape):
    results = scale_boxes_of_results_by_shape(results, input_shape, gt_shape)
    acc = f1_score(results, gt_results)
    return acc

def test_retinanet_postprocess():
    video_path = "/data/zh/LUMPI-dataset/Measurement5/cam/5/clips/5_0025.mp4"
    #imgs, inputs = convert_a_video_to_images_and_tensor(KINETICS400_DIR+'/k400_320p/_ZZswt6wHio_000121_000131.mp4', frame_num=16)
    imgs_list, imgs_tensor = convert_a_video_to_images_and_tensor(video_path, frame_num=16, new_size=(224,224))
    # run_faster_rcnn(imgs, inputs, imgs, inputs)
    # run_retinanet(imgs, inputs, imgs, inputs, show=True)
    rank = 0
    retinanet_weights, retinanet_running = load_retinanet(rank=rank)

    print(imgs_tensor.shape)

    with torch.no_grad():
        head_outputs, _, _, _ = retinanet_running(imgs_tensor, disable_post=True)

    example = torch.rand_like(imgs_tensor)
    retinanet_post = RetinanetPost(retinanet_running, example)

    results = retinanet_post.process(retinanet_running, head_outputs)

    show_detections(retinanet_weights, imgs_list, results, 0)

if __name__ == '__main__':
    video_path = "/data/zh/tmp/7_0010.mp4"
    _, imgs_tensor = convert_a_video_to_images_and_tensor(video_path, frame_num=16, frame_step=1, new_size=(224,224))

    _, gt_imgs_tensor = convert_a_video_to_images_and_tensor(video_path, frame_num=16, frame_step=1, new_size=(224,224))

    rank = 0

    print(imgs_tensor.shape)

    results = inference_faster_rcnn(imgs_tensor, rank=rank)
    gt_results = inference_faster_rcnn(gt_imgs_tensor, rank=rank) 


    frame_num = imgs_tensor.shape[0]
    for i in range(frame_num):
        visualize_faster_rcnn(imgs_tensor[i], results[i], f"/data/zh/{i}.jpg")

    for i in range(frame_num):
        visualize_faster_rcnn(gt_imgs_tensor[i], gt_results[i], f"/data/zh/gt_{i}.jpg")
    

    acc = accuracy_faster_rcnn(results, gt_results, imgs_tensor.shape[-2:], gt_imgs_tensor.shape[-2:])

    print(acc)