import torch
from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                          keypointrcnn_resnet50_fpn)

from mesh.dataset.kinetics import KINETICS400_DIR
from mesh.dnn_model.util import (convert_a_video_to_images_and_tensor,
                                 distance_accuracy, scale_keypoints, scale_keypoints_by_shape,
                                 show_keypoints, convert_tensor_to_an_image)

keypoint_rcnn_running = None
keypoint_rcnn_weights = None


def load_keypoint_rcnn(rank=0):
    score_thresh = 0.5
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(
        weights=weights, box_score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model


def run_keypoint_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False, rank=0):
    global keypoint_rcnn_running, keypoint_rcnn_weights
    if keypoint_rcnn_running is None:
        keypoint_rcnn_weights, keypoint_rcnn_running = load_keypoint_rcnn(
            rank=rank)

    with torch.no_grad():
        results = keypoint_rcnn_running(inputs)
        if online is False:
            gt_results = keypoint_rcnn_running(gt_inputs)

    if show:
        show_keypoints(imgs, results, 0)
        show_keypoints(gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_keypoints(results, inputs, gt_inputs)
        acc = distance_accuracy(results, gt_results)
        print(acc)
        return acc
    else:
        return results

def inference_keypoint_rcnn(inputs, rank=0):
    global keypoint_rcnn_running, keypoint_rcnn_weights
    if keypoint_rcnn_running is None:
        keypoint_rcnn_weights, keypoint_rcnn_running = load_keypoint_rcnn(
            rank=rank)
    with torch.no_grad():
        results = keypoint_rcnn_running(inputs)
    return results

def visualize_keypoint_rcnn(tensor_frame, result, save_path):
    img = convert_tensor_to_an_image(tensor_frame, False)
    show_keypoints([img], [result], 0, 3, save_path)

def accuracy_keypoint_rcnn(results, gt_results, input_shape, gt_shape):
    results = scale_keypoints_by_shape(results, input_shape, gt_shape)
    acc = distance_accuracy(results, gt_results)
    return acc

if __name__ == '__main__':
    video_path = "/data/zh/tmp/7_0010.mp4"
    _, imgs_tensor = convert_a_video_to_images_and_tensor(video_path, frame_num=16, frame_step=1, new_size=(224,224))
    _, gt_imgs_tensor = convert_a_video_to_images_and_tensor(video_path, frame_num=16, frame_step=1, new_size=(224,224))

    rank = 0

    print(imgs_tensor.shape)

    results = inference_keypoint_rcnn(imgs_tensor, rank=rank)
    gt_results = inference_keypoint_rcnn(gt_imgs_tensor, rank=rank) 


    frame_num = imgs_tensor.shape[0]
    for i in range(frame_num):
        visualize_keypoint_rcnn(imgs_tensor[i], results[i], f"/data/zh/{i}.jpg")

    for i in range(frame_num):
        visualize_keypoint_rcnn(gt_imgs_tensor[i], gt_results[i], f"/data/zh/gt_{i}.jpg")
    

    acc = accuracy_keypoint_rcnn(results, gt_results, imgs_tensor.shape[-2:], gt_imgs_tensor.shape[-2:])

    print(acc)