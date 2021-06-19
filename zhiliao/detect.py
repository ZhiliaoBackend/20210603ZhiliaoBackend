import os
import time
from pathlib import Path
import argparse

import torch
import torchvision
import torch.nn as nn

import cv2
import math
import numpy as np

from .enlighten.models.single_model import SingleModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = math.ceil(img_size / int(s)) * int(s)  # ceil gs-multiple
    if new_size != img_size:
        print(f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where
    # xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class PredictRes(object):

    __slots__ = ['eye_abnormal','mouth_abnormal']

    def __init__(self):
        self.eye_abnormal = None  # True if eye is closed
        self.mouth_abnormal = None  # True if mouth is opened


class Detection(object):

    __slots__ = ['device', 'model_yolo', 'model_egan', 'stride', 'img_size', 'names']

    def __init__(self):
        # Initialize
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set environment variable
        self.device = torch.device('cuda:0')

        # Load model
        ckpt = torch.load("./models/best.pt", map_location=self.device)  # load
        self.model_yolo = ckpt['model'].float().fuse().eval().half()
        opt_dict = {
            'gpu_ids':[0],
            'checkpoints_dir':'./zhiliao/enlighten/checkpoints',
            'which_model_netG':'sid_unet_resize',
            'name':'enlightening',
            'resize_or_crop':'no',
            'isTrain':False,
            'no_dropout':True,
            'no_flip':True,
            'loadSize':286,
            'fineSize':256,
            'batchSize':1,
            'input_nc':3,
            'output_nc':3,
            'vgg':0,
            'vgg_mean':None,
            'IN_vgg':None,
            'fcn':0,
            'skip':1,
            'ngf':64,
            'use_norm':1,
            'norm':'instance',
            'which_epoch':200,
            'patchD':None,
            'patchD_3':0,
            'vary':1,
            'low_times':200,
            'high_times':400,
            'noise':0,
            'input_linear':None,
            'use_wgan':0,
            'use_ragan':None,
            'hybrid_loss':None,
            'D_P_times2':None,
            'new_lr':None,
            'lr':None,
            'niter_decay':None,
            'self_attention':True,
            'syn_norm':False,
            'use_avgpool':0,
            'tanh':False,
            'times_residual':True,
            'linear_add':False,
            'linear':False,
            'latent_threshold':False,
            'latent_norm':False,
            }
        opt = argparse.Namespace(**opt_dict)
        self.model_egan = SingleModel(opt)
        self.stride = int(self.model_yolo.stride.max())  # model stride
        self.img_size = check_img_size(640, s=self.stride)  # check img_size

        # Get names
        self.names = self.model_yolo.module.names if hasattr(self.model_yolo, 'module') else self.model_yolo.names

    @torch.no_grad()
    def detect(self, img):
        """
        Detect the img
        detect(img)

        Params:
            img: numpy.array

        Returns:
            pred: PredictRes
        """
        # Padded resize
        img = letterbox(img, self.img_size, stride=self.stride)

        # Enlighten
        img = self.model_egan.predict(img)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        torch.cuda.synchronize()
        pred_array = self.model_yolo(img, False)[0]

        # Apply NMS
        pred_array = non_max_suppression(pred_array, 0.25, 0.45)
        torch.cuda.synchronize()

        pred_array = pred_array[0][:, -2:]
        pred = PredictRes()
        for confidence,idx in pred_array:
            idx = int(idx)
            if idx == 0:
                pred.eye_abnormal = False
            elif idx == 1 and pred.eye_abnormal is not True:
                pred.eye_abnormal = True
            elif idx == 2:
                pred.mouth_abnormal = True
            elif idx == 3 and pred.mouth_abnormal is not True:
                pred.mouth_abnormal = False

        return pred
