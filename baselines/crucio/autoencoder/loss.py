import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from crucio.dnn_model.faster_rcnn import load_faster_rcnn
from crucio.dnn_model.fcos import load_fcos
from crucio.dnn_model.ssd import load_ssd
from crucio.dnn_model.util import f1_score
from crucio.dnn_model.yolov5 import load_yolov5


def normalize_outputs(outputs, imgs):
    batch_size = imgs.shape[0]
    if outputs.shape[-2:] != imgs.shape[-2:]:
        resize = transforms.Resize(imgs.shape[-2:], antialias=True)
        outputs = resize(outputs)
    return batch_size, outputs


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, outputs, imgs):
        _, outputs = normalize_outputs(outputs, imgs)
        return nn.L1Loss()(outputs, imgs)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, imgs):
        _, outputs = normalize_outputs(outputs, imgs)
        return nn.MSELoss()(outputs, imgs)


class FCOSLoss(nn.Module):
    def __init__(self, rank):
        super(FCOSLoss, self).__init__()
        _, self.model = load_fcos()
        self.model = self.model.to(rank)
        self.model = DDP(self.model)

    def forward(self, outputs, imgs):
        batch_size, outputs = normalize_outputs(outputs, imgs)
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


class FasterRCNNLoss(nn.Module):
    def __init__(self, rank):
        super(FasterRCNNLoss, self).__init__()
        _, self.model = load_faster_rcnn()
        self.model = self.model.to(rank)
        self.model = DDP(self.model)

    def forward(self, outputs, imgs):
        batch_size, outputs = normalize_outputs(outputs, imgs)
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
        accuracy = f1_score(results, gt_results)
        return loss, accuracy


class SSDLoss(nn.Module):
    def __init__(self, rank):
        super(SSDLoss, self).__init__()
        _, self.model = load_ssd()
        self.model = self.model.to(rank)
        self.model = DDP(self.model)

    def forward(self, outputs, imgs):
        batch_size, outputs = normalize_outputs(outputs, imgs)
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
        # classification = loss["classification"]
        # loss = bbox_regression+classification
        loss = sum(l for l in loss.values())
        accuracy = f1_score(results, gt_results)
        return loss, accuracy


class YOLOv5Loss(nn.Module):
    def __init__(self, rank):
        super(YOLOv5Loss, self).__init__()
        # ?Import YOLOv5 sequentially to avoid following problem:
        # ?'NoneType' object is not iterable
        time.sleep(rank)
        model, nms_module, _, _ = load_yolov5()
        self.model = model.model
        for v in self.model.parameters():
            v.requires_grad = True
        self.model = self.model.to(rank)
        self.model = DDP(self.model)
        self.non_max_suppression = nms_module.non_max_suppression

    def forward(self, outputs, imgs):
        # High resolution images (imgs) must meet YOLOv5 requirements
        # (image size is multiple of max stride 32)
        # ?Call self.model only once to avoid following problems:
        # ?"RuntimeError: one of the variables needed for gradient
        # ?computation has been modified by an inplace operation"
        _, outputs = normalize_outputs(outputs, imgs)
        cat_output = self.model(torch.cat((outputs, imgs), dim=0))
        chunks = torch.chunk(cat_output[0], 2, dim=0)
        loss = torch.abs(chunks[0]-chunks[1]).mean()
        results = self.non_max_suppression(chunks[0])
        gt_results = self.non_max_suppression(chunks[1])
        accuracy = f1_score(results, gt_results, True)
        return loss, accuracy


def loss_function(index, rank=0):
    '''
    Returns loss function based on index
    Note: Weights trained by loss function 2 must be loaded
          before using loss functions 3, 4, 5 and 6
    1 -> L1Loss
    2 -> MSELoss or PSNRLoss
    3 -> FCOSLoss
    4 -> FasterRCNNLoss
    5 -> SSDLoss
    6 -> YOLOv5Loss
    '''
    if index == 1:
        if rank == 0:
            print('Using L1Loss')
        return L1Loss()
    elif index == 2:
        if rank == 0:
            print('Using MSELoss')
        return MSELoss()
    elif index == 3:
        if rank == 0:
            print('Using FCOSLoss')
        return FCOSLoss(rank)
    elif index == 4:
        if rank == 0:
            print('Using FasterRCNNLoss')
        return FasterRCNNLoss(rank)
    elif index == 5:
        if rank == 0:
            print('Using SSDLoss')
        return SSDLoss(rank)
    elif index == 6:
        if rank == 0:
            print('Using YOLOv5Loss')
        return YOLOv5Loss(rank)
    else:
        if rank == 0:
            print('Unsupported loss index!')
        exit(1)
