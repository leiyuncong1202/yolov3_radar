from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 读取检测结果CSV文件，位于det_result文件夹
def readcsv(path,img_size,x_redundancy, y_redundancy):
   img_width=img_size
   img_height=img_size
   imgname = []  # ROI图片的名字，a-b-c-d: a帧数，b列，c行，d所在小图上的ROI索引
   x1 = []  # 左上角的横坐标,基于一帧的大图
   y1 = []  # 左上角的纵坐标,基于一帧的大图
   x2 = []  # 右下角的横坐标,基于一帧的大图
   y2 = []  # 右下角的纵坐标,基于一帧的大图
   w = []  # ROI宽，基于一帧的大图
   h = []  # ROI高，基于一帧的大图
   r = []  # 置信度
   fr = []  # 帧数
   f = open(path)
   lines = f.readlines()
   for line in lines:
       imgname.append(line.split()[0])
       ################转换为基于大图的坐标########################
       # print(int(line.split()[0].split('-')[0]),int(line.split()[0].split('-')[1]), img_width, float(line.split()[1]))
       x_index = int(line.split()[0].split('-')[2])
       y_index = int(line.split()[0].split('-')[1])
       x1.append([x_index * img_width - (x_index) * x_redundancy + float(line.split()[1])])
       y1.append([y_index * img_height - (y_index) * y_redundancy + float(line.split()[2])])
       x2.append([x_index * img_width - (x_index) * x_redundancy + float(line.split()[1])+float(line.split()[5])])
       y2.append([y_index * img_height - (y_index) * y_redundancy + float(line.split()[2])+float(line.split()[6])])
       w.append([float(line.split()[5])])
       h.append([float(line.split()[6])])
       r.append([float(line.split()[7])])
       fr.append([float(line.split()[8])])

       # x1.append([int(line.split()[0].split('-')[1]) * img_size + float(line.split()[1])])
       # y1.append([int(line.split()[0].split('-')[2]) * img_size + float(line.split()[2])])
       # x2.append([int(line.split()[0].split('-')[1]) * img_size + float(line.split()[3])])
       # y2.append([int(line.split()[0].split('-')[2]) * img_size + float(line.split()[4])])
   f.close()
   detections = torch.cat((torch.Tensor(x1), torch.Tensor(y1), torch.Tensor(x2), torch.Tensor(y2), torch.Tensor(r), torch.Tensor(fr)),1)
   return detections


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    # fp.read().split("\n"):    ['ship', 'ship2333', '']
    names = fp.read().split("\n")
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#框原本是标注在416的图片上，先变为原图上
def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    #图片是先扩充成方形，然后再放缩
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    #box存储的是左上角和右下角坐标
    #截断式除法
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness 按照置信度降序排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # print(list(zip(tp,conf)))
    # Find unique classes
    unique_classes = np.unique(target_cls)
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # 对于某一类
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
        print('n_p: ',n_p)
        print('n_gt: ',n_gt)
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])
            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])
            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    pd=r
    pf=1-p
    f1 = 2 * p * r / (p + r + 1e-16)
    print('precision: ',p)
    print('recall: ',r)
    print('ap: ',ap)
    print('f1',f1)
    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # plt.plot(mrec,mpre)      # 绘制PR曲线
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.show()
    # plt.savefig("PR.png")

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]       # np.where()的返回值为元组形式。元组的每个元素表示每个维度上的坐标
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    # 对于每张图片
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        # pred_labels = output[:, -2]  # 层级索引
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])  # TP: IoU>0.5的检测框数量（同一Ground Truth只计算一次）
        annotations = targets[targets[:, 0] == sample_i][:, 1:]   # 一张图片上所有的target信息：[cls,x,y,x,y]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break
                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                # 对于每一个pred_box,与所有GT计算IoU,求最大值
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        # 三个变量的size相等，都是一张图片上pred_box的数目
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics



#求GT与anchor的IoU
#从3个anchor里筛选出最符合的那个
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


#求box1与box2的交并比
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    #clamp将输入input张量每个元素的夹紧到区间 [min,max]，输出为Tensor
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


#NMS
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.35):
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        if not image_pred.size(0):
            continue
        score = image_pred[:, 4]*image_pred[:,5:].max(1)[0]
        # Sort by it      #argsort():返回数组从小到大的索引值
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # class_confs, class_preds = image_pred[:, 5:-1].max(1, keepdim=True)      # 层级索引
        # detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(),image_pred[:,-1:]), 1)  # 层级索引
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # label_match = detections[0, -2] == detections[:, -2]     # 层级索引
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match
            #应该去除的那些格子的objectness
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


# 对于一帧图片做全局NMS，无conf筛选
def global_nms(image_pred,nms_thres=0.35):
    # print(image_pred[0])
    output = [None for _ in range(len(image_pred))]
    score = image_pred[:, 4]*image_pred[:,5:].max(1)[0]
    # Sort by it      #argsort():返回数组从小到大的索引值
    image_pred = image_pred[(-score).argsort()]
    detections = image_pred
    # Perform non-maximum suppression
    keep_boxes = []
    while detections.size(0):
        large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
        # label_match = detections[0, -2] == detections[:, -2]
        # invalid = large_overlap & label_match
        keep_boxes += [detections[0]]
        detections = detections[~large_overlap]
    if keep_boxes:
        output = torch.stack(keep_boxes)
    # print(output[0])
    return output


def origin_nms(prediction, conf_thres=0.5, nms_thres=0.4):
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    #len(prediction)为batchsize。即对于每张图片清空一次
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        #当一张图片上不含有检测框时
        if not image_pred.size(0):
            continue
        score = image_pred[:, 4]*image_pred[:,5:].max(1)[0]
        # Sort by it      #argsort():返回数组从小到大的索引值
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:-1].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(),image_pred[:,-1:]), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            #score最大的一个box和其他所有box求iou
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            #当一个box的类别和score最大的一个box的类别 相同时，其对应的label_match元素为1
            label_match = detections[0, -2] == detections[:, -2]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            #和score最高的box的同一类的box里，IOU超过阈值的去除。invalid中对应元素为1的是应该去除的
            invalid = large_overlap & label_match
            # #应该去除的那些格子的objectness
            # weights = detections[invalid, 4:5]
            # # Merge overlapping bboxes by order of confidence
            # detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def soft_nms_gaussian(prediction, score_threshold=0.001, sigma=0.5, top_k=-1):
    """Soft NMS implementation.
    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
    Args:
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    """
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf = image_pred[:, 4]
        class_confs, class_preds = image_pred[:, 5:-1].max(1, keepdim=True)
        # 只有一个分类，只参考conf
        score = conf * image_pred[:,5:].max(1)[0]
        # # 有多个分类时
        # score = conf * class_confs
        image_pred = image_pred[(-score).argsort()]
        box_scores = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(), image_pred[:, -1:]), 1)
        picked_box_scores = []
        while box_scores.size(0) > 0:
            max_score_index = torch.argmax(box_scores[:, 4])
            cur_box_prob = box_scores[max_score_index, :].clone()
            picked_box_scores.append(cur_box_prob)
            if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
                break
            cur_box = cur_box_prob[:4]
            box_scores[max_score_index, :] = box_scores[-1, :]
            box_scores = box_scores[:-1, :]
            ious = bbox_iou(cur_box.unsqueeze(0), box_scores[:, :4])
            box_scores[:, 4] = box_scores[:, 4] * torch.exp(-(ious * ious) / sigma)        # gaussian soft_nms
            box_scores = box_scores[box_scores[:, 4] > score_threshold, :]
        if len(picked_box_scores) > 0:
            output[image_i] = torch.stack(picked_box_scores)

    return output


# 预处理GT框
# pred_boxes:特征图尺度上中心点坐标和宽高；pred_cls:[0,1]之间的类概率；
# target：调整过尺度的GT，6维 （box的序号，label，GT），GT坐标为416*416上的相对值；
# anchors:特征图尺度上的anchor对；ignore_thres:objectness小于此值时，忽略
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    #batchsize
    nB = pred_boxes.size(0)
    #anchor数目，3
    nA = pred_boxes.size(1)
    #类数目
    nC = pred_cls.size(-1)
    #格子数目，如13
    nG = pred_boxes.size(2)

    # Output tensors
    #fill_ :用某值填充Tensor
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    #对于每个格子是每个类的概率
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    # 相对值*13
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    #stack()默认dim=0
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    #best_n:0,1,2
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    gi = torch.clamp(gi, 0, noobj_mask.size()[2] - 1)
    gj = torch.clamp(gj, 0, noobj_mask.size()[3] - 1)
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    #t():tensor转置
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


