import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from mesh.dnn_model.instance_segmentation import load_mask_rcnn
from mesh.dnn_model.keypoint_detection import load_keypoint_rcnn
from mesh.dnn_model.object_detection import load_faster_rcnn, load_retinanet
from mesh.dnn_model.semantic_segmentation import load_deeplabv3, load_fcn
from mesh.dnn_model.util import (category_accuracy, distance_accuracy,
                                 f1_score, miou_accuracy, pixel_accuracy)
from mesh.dnn_model.video_classification import (load_video_resnet,
                                                 load_video_s3d)
from mesh.msc.utils import DDP_PARAMETER_CHECK


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, imgs):
        return nn.MSELoss(reduction="none")(outputs, imgs)


class FasterRCNNLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(FasterRCNNLoss, self).__init__()
        _, self.model = load_faster_rcnn()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        self.model.eval()
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results[:batch_size]
        gt_results = cat_results[batch_size:]
        self.model.train()
        for dictionary in gt_results:
            del dictionary['scores']
        loss = self.model(images=outputs, targets=gt_results)
        # loss_box_reg = loss["loss_box_reg"]
        # loss_classifier = loss["loss_classifier"]
        # loss_objectness = loss["loss_objectness"]
        # loss_rpn_box_reg = loss["loss_rpn_box_reg"]
        # loss = loss_box_reg+loss_classifier+loss_objectness+loss_rpn_box_reg
        loss = sum(l for l in loss.values())
        # models/detection/roi_heads.py/fastrcnn_loss
        accuracy = f1_score(results, gt_results)
        return loss, accuracy


class RetinaNetLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(RetinaNetLoss, self).__init__()
        _, self.model = load_retinanet()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        self.model.eval()
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results[:batch_size]
        gt_results = cat_results[batch_size:]
        self.model.train()
        for dictionary in gt_results:
            del dictionary['scores']
        loss = self.model(images=outputs, targets=gt_results)
        # bbox_regression = loss["bbox_regression"]
        # bbox_ctrness = loss["bbox_ctrness"]
        # classification = loss["classification"]
        # loss = bbox_regression+bbox_ctrness+classification
        loss = sum(l for l in loss.values())
        accuracy = f1_score(results, gt_results)
        return loss, accuracy


class MaskRCNNLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(MaskRCNNLoss, self).__init__()
        _, self.model = load_mask_rcnn()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        self.model.eval()
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results[:batch_size]
        gt_results = cat_results[batch_size:]
        self.model.train()
        for dictionary in gt_results:
            dictionary['masks'] = dictionary['masks'].squeeze(1)
            del dictionary['scores']
        loss = self.model(images=outputs, targets=gt_results)
        # loss_box_reg = loss["loss_box_reg"]
        # loss_classifier = loss["loss_classifier"]
        # loss_objectness = loss["loss_objectness"]
        # loss_rpn_box_reg = loss["loss_rpn_box_reg"]
        # loss_mask = loss["loss_mask"]
        # loss = loss_box_reg+loss_classifier+loss_objectness+loss_rpn_box_reg+loss_mask
        loss = sum(l for l in loss.values())
        # models/detection/roi_heads.py/maskrcnn_loss
        accuracy = miou_accuracy(results, gt_results)
        return loss, accuracy


class FCNLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(FCNLoss, self).__init__()
        _, self.model = load_fcn()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results['out'][:batch_size]
        gt_results = cat_results['out'][batch_size:]
        abs_diff = torch.abs(results-gt_results)
        loss = torch.mean(abs_diff)
        accuracy = pixel_accuracy(results, gt_results)
        return loss, accuracy


class DeepLabV3Loss(nn.Module):
    def __init__(self, world_size, rank):
        super(DeepLabV3Loss, self).__init__()
        _, self.model = load_deeplabv3()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results['out'][:batch_size]
        gt_results = cat_results['out'][batch_size:]
        abs_diff = torch.abs(results-gt_results)
        loss = torch.mean(abs_diff)
        accuracy = pixel_accuracy(results, gt_results)
        return loss, accuracy


class KeypointRCNNLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(KeypointRCNNLoss, self).__init__()
        _, self.model = load_keypoint_rcnn()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        self.model.eval()
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results[:batch_size]
        gt_results = cat_results[batch_size:]
        self.model.train()
        for dictionary in gt_results:
            del dictionary['scores']
        loss = self.model(images=outputs, targets=gt_results)
        # loss_box_reg = loss["loss_box_reg"]
        # loss_classifier = loss["loss_classifier"]
        # loss_objectness = loss["loss_objectness"]
        # loss_rpn_box_reg = loss["loss_rpn_box_reg"]
        # loss_keypoint = loss["loss_keypoint"]
        # loss = loss_box_reg+loss_classifier+loss_objectness+loss_rpn_box_reg+loss_keypoint
        loss = sum(l for l in loss.values())
        # models/detection/roi_heads.py/keypointrcnn_loss
        accuracy = distance_accuracy(results, gt_results)
        return loss, accuracy


class VideoResnetLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(VideoResnetLoss, self).__init__()
        self.weights, self.model = load_video_resnet()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results[:batch_size]
        gt_results = cat_results[batch_size:]
        abs_diff = torch.abs(results-gt_results)
        loss = torch.mean(abs_diff)
        accuracy = category_accuracy(self.weights, results, gt_results)
        return loss, accuracy


class VideoS3DLoss(nn.Module):
    def __init__(self, world_size, rank):
        super(VideoS3DLoss, self).__init__()
        self.weights, self.model = load_video_s3d()
        self.model = self.model.to(rank)
        if world_size > 1:
            self.model = DDP(
                self.model, find_unused_parameters=DDP_PARAMETER_CHECK)

    def forward(self, outputs, imgs):
        batch_size = imgs.shape[0]
        with torch.no_grad():
            cat_results = self.model(torch.cat((outputs, imgs), dim=0))
        results = cat_results[:batch_size]
        gt_results = cat_results[batch_size:]
        abs_diff = torch.abs(results-gt_results)
        loss = torch.mean(abs_diff)
        accuracy = category_accuracy(self.weights, results, gt_results)
        return loss, accuracy


def loss_function(index, world_size, rank=0):
    '''
    Returns loss function based on index
    Note: Weights trained by loss function 1 must be loaded
          before using other loss functions
    1 -> MSELoss
    2 -> FasterRCNNLoss
    3 -> RetinaNetLoss
    4 -> MaskRCNNLoss
    5 -> FCNLoss
    6 -> DeepLabV3Loss
    7 -> KeypointRCNNLoss
    8 -> VideoResnetLoss
    9 -> VideoS3DLoss
    '''
    if index == 1:
        print('Using (1)MSELoss')
        return MSELoss()
    elif index == 2:
        print('Using (2)FasterRCNNLoss')
        return FasterRCNNLoss(world_size, rank)
    elif index == 3:
        print('Using (3)RetinaNetLoss')
        return RetinaNetLoss(world_size, rank)
    elif index == 4:
        print('Using (4)MaskRCNNLoss')
        return MaskRCNNLoss(world_size, rank)
    elif index == 5:
        print('Using (5)FCNLoss')
        return FCNLoss(world_size, rank)
    elif index == 6:
        print('Using (6)DeepLabV3Loss')
        return DeepLabV3Loss(world_size, rank)
    elif index == 7:
        print('Using (7)KeypointRCNNLoss')
        return KeypointRCNNLoss(world_size, rank)
    elif index == 8:
        print('Using (8)VideoResnetLoss')
        return VideoResnetLoss(world_size, rank)
    elif index == 9:
        print('Using (9)VideoS3DLoss')
        return VideoS3DLoss(world_size, rank)
    else:
        print('Unsupported loss index!')
        exit(1)
