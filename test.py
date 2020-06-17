# -*- coding: utf-8 -*-
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import shutil
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from ptflops import get_model_complexity_info

os.environ["CUDA_VISIBLE_DEVICES"]='3'

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn
    )
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    #tqdm:进度�?
    for batch_i, (img_paths, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Extract abstract coor
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = model(input_imgs)    # [8, 10647, 7]
            # outputs = origin_nms(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # print(outputs[0])
            # conf_test = []
            # for output in outputs:
            #     if output is not None:
            #         conf_test += output[:, -2].tolist()
            # print(conf_test[:20])

            # outputs = soft_nms_gaussian(outputs)
            # level_index = []         # 层级结构
            # for output in outputs:
            #     if output is not None:
            #         level_index += output[:,-1].tolist()
            # print(level_index)

             # Compute true positives, predicted scores and predicted labels per sample
             # TP: IoU>0.5的检测框数量（同一Ground Truth只计算一次）
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        # print(sample_metrics)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_cv_multi.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/0601.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="/home/detection/projects/PyTorch-YOLOv3/weights/yolov3_cv_multi_0601_cv.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/multi_ship.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument('--result', type=str, default=None, help="FLOPs and model size")
    parser.add_argument("--det_folder", type=str, default="data/0315data/detResult/", help="path to det result") # 检测结果的文件位置。每帧的所有小图的检测结果存放于一个文件夹�?
    opt = parser.parse_args()
    print(opt)

    torch.cuda.synchronize()
    start = time.time()
    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]     # txt文件路径
    class_names = load_classes(data_config["names"])        # 类名数组

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights  Darkent的格�?
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights�?pth�?只保存了参数
        model.load_state_dict(torch.load(opt.weights_path))
        # # 保存了模型和参数
        # model=torch.load(opt.weights_path)



    # 计算FLOPs与model size
    # if opt.result is None:
    #     ost = sys.stdout
    # else:
    #     ost = open(opt.result, 'w')
    # flops, params = get_model_complexity_info(model, (3, 416, 416),
    #                                           as_strings=True,
    #                                           print_per_layer_stat=True,
    #                                           ost=ost)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("Compute mAP...")
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    torch.cuda.synchronize()
    end = time.time()
    print(f"total time: {end - start}")  # 包含加载数据，包含加载模�?    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
    print(f"precision: {precision.mean()}")
    print(f"recall: {recall.mean()}")
    print(f"mAP: {AP.mean()}")