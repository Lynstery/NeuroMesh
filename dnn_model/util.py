import ffmpeg
import numpy as np
import torch
from PIL import Image, ImageFont
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.models.detection.transform import (resize_boxes,
                                                    resize_keypoints)
from torchvision.ops import box_iou
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import (draw_bounding_boxes, draw_keypoints,
                               draw_segmentation_masks)
import lzma
import pickle
import os

def save_tensor_to_video(video_path, tensor):
    '''
    Saves a tensor (no batch dimension) to specified video directory
    video_path -> Absolute path to video directory
    tensor -> A tensor from decoder
    is_yuv -> Whether tensor is in YUV color space
    '''
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    len = tensor.shape[1]
    for _ in range(len):
        img = convert_tensor_to_an_image(tensor[:, _])
        str_num = "{:04d}".format(_+1)
        img.save(os.path.join(video_path, 'frame'+str_num+'.png'))

        
def convert_a_video_to_images_and_tensor(video_path, frame_num=5, frame_step=10, new_size=None, video_tensor=False, rank=0):
    probe = ffmpeg.probe(video_path)
    duration = int(probe['streams'][0]['nb_frames'])
    if frame_num > duration:
        frame_num = duration
    if frame_num * frame_step > duration:
        frame_step = duration // frame_num

    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('select', 'not(mod(n,{0}))'.format(frame_step))
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=frame_num)
        .run(capture_stdout=True, quiet=True)
    )

    video_data = np.frombuffer(out, np.uint8).reshape(
        [-1, int(probe['streams'][0]['height']), int(probe['streams'][0]['width']), 3])
    assert video_data.shape[0] == frame_num
    sampled_imgs = [Image.fromarray(video_data[vid, :, :, :]).convert(
        'RGB') for vid in range(frame_num)]
    if not video_data.flags.writeable:
        video_data = video_data.copy()
    tensor = torch.from_numpy(video_data).permute(0, 3, 1, 2).float() / 255.0

    if new_size:
        resize = transforms.Resize(new_size, antialias=True)
        sampled_imgs = [resize(_) for _ in sampled_imgs]
        tensor = resize(tensor)
    if video_tensor:
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
    tensor = tensor.to(rank)
    return sampled_imgs, tensor


def tensor_normalization(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    tensor = (tensor - min_value) / (max_value - min_value + 1e-6)
    return tensor

def convert_image_to_tensor(img_path, is_yuv, is_gpu, size=None, rank=0):
    '''
    Convert an RGB image to a tensor (no batch dimension)
    img_path -> Image absolute path 
    is_yuv -> Whether to use YUV color space
    is_gpu -> Whether to move tensor to GPU
    size -> List [H,W] for image resize
    '''
    if is_yuv:
        tensor_mode = 'YCbCr'
    else:
        tensor_mode = 'RGB'
    image = Image.open(img_path).convert(tensor_mode)
    tensor = transforms.ToTensor()(image)
    if is_gpu:
        tensor = tensor.to(rank)
    if size:
        tensor = transforms.Resize(size, antialias=True)(tensor)
    return tensor


def convert_pil_image_to_tensor(img, is_yuv, is_gpu, size=None, rank=0):
    if is_yuv:
        tensor_mode = 'YCbCr'
    else:
        tensor_mode = 'RGB'
    image = img.convert(tensor_mode)
    tensor = transforms.ToTensor()(image)
    if is_gpu:
        tensor = tensor.to(rank)
    if size:
        tensor = transforms.Resize(size, antialias=True)(tensor)
    return tensor

def convert_tensor_to_an_image(img, show=True):
    to_pil = ToPILImage()
    img = to_pil(img)
    if show:
        img.show()
    return img

def save_compressed_data_at(data_path, compressed_data):
    '''
    data_path -> Absolute path to save compressed data
    compressed_data -> Compressed data from encoder (tensor)
    Return absolute path to compressed data
    '''
    compressed_data = lzma.compress(pickle.dumps(compressed_data))
    with open(data_path, 'wb') as f:
        f.write(compressed_data)
    return data_path

def load_compressed_data(data_path, rank=None):
    '''
    data_path -> Absolute path of compressed data
    Return compressed data from encoder (tensor)
    '''
    with open(data_path, 'rb') as f:
        compressed_data = f.read()
    compressed_data = pickle.loads(lzma.decompress(compressed_data))
    if rank is not None:
        compressed_data = compressed_data.to(rank)
    return compressed_data


def get_folder_size(folder_path):
    '''
    Calculates total size of all files in specified folder (KB)
    '''
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size/1024

def convert_a_video_tensor_to_images(tensor, frame_index, show=True):
    imgs = tensor.permute(1, 0, 2, 3)
    frame_num = imgs.shape[0]
    if frame_index >= frame_num:
        frame_index = frame_num-1
    return convert_tensor_to_an_image(imgs[frame_index], show)

def convert_a_video_tensor_to_images_list(tensor):
    imgs = tensor.permute(1, 0, 2, 3)
    frame_num = imgs.shape[0]
    pil_imgs_list = []
    for i in range(0, frame_num):
        pil_img = convert_tensor_to_an_image(imgs[i], False)
        pil_imgs_list.append(pil_img)

    return pil_imgs_list
        

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


def show_detections(weights_or_labels, imgs, results, index, is_yolov5=False, save=None):
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
    img = imgs[index]
    if isinstance(img, str):
        tensor = read_image(img)
    elif isinstance(img, Image.Image):
        tensor = pil_to_tensor(img)
    boxes = draw_bounding_boxes(
        tensor, boxes, names, width=4, font='/usr/share/fonts/truetype/freefont/FreeSans.ttf', font_size=22)
    img = to_pil_image(boxes)
    if save is None:
        img.show()
    else:
        img.save(save)


def show_segmentations(imgs, results, index, proba_thre=0.5, save=None):
    masks = (results[index]['masks'] > proba_thre).squeeze(1)
    img = imgs[index]
    if isinstance(img, str):
        tensor = read_image(img)
    elif isinstance(img, Image.Image):
        tensor = pil_to_tensor(img)
    segmentation = draw_segmentation_masks(tensor, masks=masks, alpha=0.9)
    img = to_pil_image(segmentation)
    if save is not None:
        img.save(save)
    else:
        img.show()


def show_masks(weights, results, index=0, save=None):
    prediction = results["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(
        weights.meta["categories"])}
    mask = normalized_masks[index, class_to_idx["person"]]
    img = to_pil_image(mask)
    if save is not None:
        img.save(save)
    else:
        img.show()


def show_keypoints(imgs, results, index, radius=3, save=None):
    img = imgs[index]
    if isinstance(img, str):
        tensor = read_image(img)
    elif isinstance(img, Image.Image):
        tensor = pil_to_tensor(img)
    keypoints = draw_keypoints(
        tensor, keypoints=results[index]['keypoints'], colors="blue", radius=radius)
    img = to_pil_image(keypoints)
    if save is not None:
        img.save(save)
    else:
        img.show()

def scale_boxes_of_results_by_shape(results, image_shapes, target_image_sizes, is_yolov5=False):
    '''
    Scale bounding boxes of inference results based on target images
    results -> A batch of inference results
    inputs -> A batch of input images
    target -> A batch of target images
    is_yolov5 -> Whether analytics task is YOLOv5
    '''
    if image_shapes != target_image_sizes:
        for i in range(len(results)):
            boxes, _ = get_boxes_and_labels(results, i, is_yolov5)
            boxes = resize_boxes(boxes, image_shapes, target_image_sizes)
            if is_yolov5:
                results[i][:, 0:4] = boxes
            else:
                results[i]["boxes"] = boxes
    return results

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

def scale_segmentation_by_shape(results, image_shapes, target_image_sizes):
    if image_shapes != target_image_sizes:
        for i in range(len(results)):
            boxes = resize_boxes(results[i]["boxes"], image_shapes, target_image_sizes)
            results[i]["boxes"] = boxes
            masks = results[i]['masks']
            resize = transforms.Resize(target_image_sizes, antialias=True)
            results[i]['masks'] = resize(masks)
    return results

def scale_segmentations(results, inputs, target):
    image_shapes = inputs.shape[-2:]
    target_image_sizes = target.shape[-2:]
    if image_shapes != target_image_sizes:
        for i in range(len(results)):
            boxes = resize_boxes(results[i]["boxes"], image_shapes, target_image_sizes)
            results[i]["boxes"] = boxes
            masks = results[i]['masks']
            resize = transforms.Resize(target_image_sizes, antialias=True)
            results[i]['masks'] = resize(masks)
    return results

def scale_masks_by_shape(results, image_shapes, target_image_sizes):
    if image_shapes != target_image_sizes:
        out = results['out']
        aux = results['aux']
        resize = transforms.Resize(target_image_sizes, antialias=True)
        results['out'] = resize(out)
        results['aux'] = resize(aux)
    return results

def scale_masks(results, inputs, target):
    image_shapes = inputs.shape[-2:]
    target_image_sizes = target.shape[-2:]
    if image_shapes != target_image_sizes:
        out = results['out']
        aux = results['aux']
        resize = transforms.Resize(target_image_sizes, antialias=True)
        results['out'] = resize(out)
        results['aux'] = resize(aux)
    return results

def scale_keypoints_by_shape(results, image_shapes, target_image_sizes):
    if image_shapes != target_image_sizes:
        for i in range(len(results)):
            boxes = resize_boxes(
                results[i]["boxes"], image_shapes, target_image_sizes)
            results[i]["boxes"] = boxes
            keypoints = resize_keypoints(
                results[i]['keypoints'], image_shapes, target_image_sizes)
            results[i]['keypoints'] = keypoints
    return results

def scale_keypoints(results, inputs, target):
    image_shapes = inputs.shape[-2:]
    target_image_sizes = target.shape[-2:]
    if image_shapes != target_image_sizes:
        for i in range(len(results)):
            boxes = resize_boxes(
                results[i]["boxes"], image_shapes, target_image_sizes)
            results[i]["boxes"] = boxes
            keypoints = resize_keypoints(
                results[i]['keypoints'], image_shapes, target_image_sizes)
            results[i]['keypoints'] = keypoints
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
                biou = box_iou(boxes, gt_boxes)
                best_match = torch.nonzero(biou >= min_iou).tolist()
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


def miou_accuracy(results, gt_results, min_iou=0.7, proba_thres=0.5):
    assert len(results) == len(gt_results)
    number = len(gt_results)
    miou = []
    for i in range(number):
        boxes = results[i]['boxes']
        labels = results[i]['labels']
        masks = (results[i]['masks'] > proba_thres).squeeze(1)
        gt_boxes = gt_results[i]['boxes']
        gt_labels = gt_results[i]['labels']
        gt_masks = (gt_results[i]['masks'] > proba_thres).squeeze(1)
        instance = gt_masks.shape[0]
        iou = 0
        if instance == 0:
            iou = 1
        else:
            biou = box_iou(boxes, gt_boxes)
            best_match = torch.nonzero(biou >= min_iou).tolist()
            for j in range(len(best_match)):
                index = best_match[j][0]
                gt_index = best_match[j][1]
                if labels[index] == gt_labels[gt_index]:
                    intersection = torch.logical_and(
                        masks[index], gt_masks[gt_index]).sum().item()
                    union = torch.logical_or(
                        masks[index], gt_masks[gt_index]).sum().item()
                    assert union != 0
                    iou += (intersection/union)
            iou /= instance
        miou.append(iou)
    return np.mean(miou)


def pixel_accuracy(results, gt_results):
    if isinstance(results, dict):
        out = results['out']
    else:
        out = results
    if isinstance(gt_results, dict):
        gt_out = gt_results['out']
    else:
        gt_out = gt_results
    assert out.shape == gt_out.shape
    out = out.softmax(dim=1)
    _, max_indices = torch.max(out, dim=1, keepdim=True)
    binary_out = torch.zeros_like(out, device=out.device)
    binary_out[max_indices == torch.arange(
        out.shape[1], device=out.device).reshape(1, -1, 1, 1)] = 1
    gt_out = gt_out.softmax(dim=1)
    _, gt_max_indices = torch.max(gt_out, dim=1, keepdim=True)
    gt_binary_out = torch.zeros_like(gt_out, device=out.device)
    gt_binary_out[gt_max_indices == torch.arange(
        gt_out.shape[1], device=out.device).reshape(1, -1, 1, 1)] = 1
    pixel_num = gt_binary_out.shape[0] * \
        gt_binary_out.shape[2]*gt_binary_out.shape[3]
    binary_out = binary_out.view(pixel_num, binary_out.shape[1])
    gt_binary_out = gt_binary_out.view(pixel_num, gt_binary_out.shape[1])
    same_pixels = torch.sum(
        torch.all(torch.eq(binary_out, gt_binary_out), dim=1)).item()
    return same_pixels/pixel_num


def distance_accuracy(results, gt_results, min_iou=0.7, dis_thres=10):
    assert len(results) == len(gt_results)
    number = len(gt_results)
    dis_acc = []
    for i in range(number):
        boxes = results[i]['boxes']
        labels = results[i]['labels']
        keypoints = results[i]['keypoints']
        gt_boxes = gt_results[i]['boxes']
        gt_labels = gt_results[i]['labels']
        gt_keypoints = gt_results[i]['keypoints']
        instance = gt_keypoints.shape[0]
        dis = 0
        if instance == 0:
            dis = 1
        else:
            biou = box_iou(boxes, gt_boxes)
            best_match = torch.nonzero(biou >= min_iou).tolist()
            for j in range(len(best_match)):
                index = best_match[j][0]
                gt_index = best_match[j][1]
                if labels[index] == gt_labels[gt_index]:
                    distances = torch.norm(
                        keypoints[index, ..., :-1] - gt_keypoints[gt_index, ..., :-1], dim=1)
                    count = (distances < dis_thres).sum().item()
                    dis += (count/len(distances))
            dis /= instance
        dis_acc.append(dis)
    return np.mean(dis_acc)


def get_categories(weights, results, show=True):
    prediction = results.squeeze(0).softmax(0)
    label = prediction.argmax().item()
    score = prediction[label].item()
    category = weights.meta["categories"][label]
    if show:
        print(f"{category}: {100 * score}%")
    return label, category, score


def category_accuracy(weights, results, gt_results, show=False):
    category, score = get_categories(weights, results, show)
    gt_category, gt_score = get_categories(weights, gt_results, show)
    acc = 0
    if category == gt_category:
        if gt_score != 0:
            acc = 1
            if score < gt_score:
                acc = score/gt_score
    return acc
