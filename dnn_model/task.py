from mesh.dnn_model.object_detection import inference_faster_rcnn, inference_retinanet, visualize_faster_rcnn, visualize_retinanet, accuracy_faster_rcnn, accuracy_retinanet
from mesh.dnn_model.instance_segmentation import inference_mask_rcnn, visualize_mask_rcnn, accuracy_mask_rcnn
from mesh.dnn_model.keypoint_detection import inference_keypoint_rcnn, visualize_keypoint_rcnn, accuracy_keypoint_rcnn
from mesh.dnn_model.semantic_segmentation import inference_fcn, inference_deeplabv3, visualize_fcn, visualize_deeplabv3, accuracy_fcn, accuracy_deeplabv3

def run_inference(model_name, inputs, rank=0):
    if model_name == "faster_rcnn":
        return inference_faster_rcnn(inputs, rank=rank)
    elif model_name == "retinanet":
        return inference_retinanet(inputs, rank=rank)
    elif model_name == "mask_rcnn":
        return inference_mask_rcnn(inputs, rank=rank)
    elif model_name == "keypoint_rcnn":
        return inference_keypoint_rcnn(inputs, rank=rank)
    elif model_name == "fcn":
        return inference_fcn(inputs, rank=rank)
    elif model_name == "deeplabv3":
        return inference_deeplabv3(inputs, rank=rank)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

def visualize_result(model_name, tensor_frame, result, save_path):
    if model_name == "faster_rcnn":
        visualize_faster_rcnn(tensor_frame, result, save_path)
    elif model_name == "retinanet":
        return visualize_retinanet(tensor_frame, result, save_path)
    elif model_name == "mask_rcnn":
        visualize_mask_rcnn(tensor_frame, result, save_path)
    elif model_name == "keypoint_rcnn":
        visualize_keypoint_rcnn(tensor_frame, result, save_path)
    elif model_name == "fcn":
        visualize_fcn(tensor_frame, result, save_path) 
    elif model_name == "deeplabv3":
        visualize_deeplabv3(tensor_frame, result, save_path)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

def calculate_accuracy(model_name, results, gt_results, input_shape, gt_shape):
    if model_name == "faster_rcnn":
        return accuracy_faster_rcnn(results, gt_results, input_shape, gt_shape)
    elif model_name == "retinanet":
        return accuracy_retinanet(results, gt_results, input_shape, gt_shape)
    elif model_name == "mask_rcnn":
        return accuracy_mask_rcnn(results, gt_results, input_shape, gt_shape)
    elif model_name == "keypoint_rcnn":
        return accuracy_keypoint_rcnn(results, gt_results, input_shape, gt_shape)
    elif model_name == "fcn":
        return accuracy_fcn(results, gt_results, input_shape, gt_shape)
    elif model_name == "deeplabv3":
        return accuracy_deeplabv3(results, gt_results, input_shape, gt_shape)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
