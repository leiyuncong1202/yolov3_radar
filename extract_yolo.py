from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from yolov3_feat.extract_feat import extract_feat

import os
import sys
import time
import datetime
import argparse
import shutil

from PIL import Image
import re
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 PyTorch')
    parser.add_argument("--image_folder", type=str, default="data/1114data/images/", help="path to dataset") # 原图片位置。每帧的所有小图存放于一个文件夹下，该文件夹命名为帧号
    parser.add_argument("--model_def", type=str, default="config/yolov3-voc.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_1118_final.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/radar3data/ship.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.05, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.01, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches") # 检测时的batch_size,VGG的batch_size需在vgg_test.py中调整
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--det_folder", type=str, default="data/1114data/detResult/", help="path to detection results file")  # 检测结果的文件位置。每帧的所有小图的检测结果存放于一个文件夹下
    parser.add_argument("--out_folder", type=str, default="data/1114data/output/", help="path to detection results image")
    parser.add_argument("--roi_folder", type=str, default="data/1114data/ROI/", help="path to detection ROI image")
    parser.add_argument('--csv_folder', default='data/1114data/csv/', help='path to csv_result')  # 最终csv形式的输出
    parser.add_argument('--mat_folder', default='mat/mat0.01/', help='path to mat_result')  # 最终mat形式的输出
    parser.add_argument('--start_frame', type=int, default=1, help='start-frame')  # 开始帧
    parser.add_argument('--end_frame', type=int, default=1056, help='end-frame')  # 结束帧
    parser.add_argument('--xpart', type=int, default=3, help="xpart")  # 横向分几个
    parser.add_argument('--ypart', type=int, default=4, help="ypart")  # 纵向分几个
    opt = parser.parse_args()
    return opt


##############################记录检测结果，每张图片对应一个csv文件################################
def draw_detections_txt(imgs, img_detections, det_folder, img_size):
    if os.path.exists(det_folder):
        shutil.rmtree(det_folder)
        os.mkdir(det_folder)

    for (path, detections) in zip(imgs, img_detections):
        img_name = path.split('/')[-1].split('.')[0]  # 帧号-列号-行号
        frame_name = img_name.split('-')[0]  # 图片所属帧号
        det_result = det_folder + frame_name + ".csv"
        # print(det_result)
        imgname = [] # 图片名称
        bx = []  # ROI左上角的横坐标,基于一帧的大图
        by = []  # ROI左上角的纵坐标，基于一帧的大图
        x = []  # 同bx, 左上角的横坐标
        y = []  # 同by, 左上角的纵坐标
        w = []  # ROI宽，基于一帧的大图
        h = []  # ROI高，基于一帧的大图
        r = []  # 置信度
        fr = []  # 帧数
        if detections is not None:
            # print('记录了%s个预测框' % (detections.numpy().shape[0]))
            for var in (detections.numpy()):
                imgname.append(img_name)
                bx.append(max(0,var[0]))
                by.append(max(0,var[1]))
                x.append(max(0,var[0]))
                y.append(max(0,var[1]))
                w.append(max(0,var[2] - var[0]))
                h.append(max(0,var[3] - var[1]))
                r.append(var[5]*100)
                fr.append(frame_name)
            df = pd.DataFrame({'imgname': imgname, 'bx': bx, 'by': by, 'x': x, 'y': y, 'w': w, 'h': h, 'r': r, 'fr': fr})
            df.to_csv(det_result, sep=' ', mode='a', header=False, index=0)
        else:
            if not os.path.exists(det_result):
                os.mknod(det_result)


#########################在图片上绘制检测框并保存##########################################
def draw_detections(imgs, img_detections, out_folder):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)

    #Bounding-box colors，使用matplotlib量化离散化的色图
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # print(img.shape)
        # print(img.size)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            #img.shape[:2]输出前两维，即原始高度和宽度
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            #一张图上有多个框时，从colors随机提取出n_cls_preds个元素
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                #获取一个位置的值，并将其作为python的数据类型
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label图片上的标注
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
        # Save generated image with detections
        #不显示坐标尺寸
        plt.axis("off")
        #删除坐标轴的刻度显示
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        #保存pad为0的紧框，即只有该图片
        plt.savefig(out_folder + f"/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()


def main():
    opt = parse_args()
    print(opt)
    prev_time = time.time()  # # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    # ###############################预测图片与预测结果的存放位置#############################
    if not os.path.exists(opt.out_folder):
        os.mkdir(opt.out_folder)
    if not os.path.exists(opt.det_folder):
        os.mkdir(opt.det_folder)
    if not os.path.exists(opt.roi_folder):
        os.mkdir(opt.roi_folder)
    print("\n")
    print("*" * 50)
    print("开始检测")
    print("*" * 50)
    ###############################################加载数据##############################
    dataloader = DataLoader(
        ImageFolder(opt.image_folder),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    ##################################加载模型和权重####################################
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # # 单机多卡
    # device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    # model = Darknet(opt.model_def, img_size=opt.img_size)
    # model = nn.DataParallel(model).to(device)

    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        #单卡
        model.load_state_dict(torch.load(opt.weights_path))
        # #多卡
        # state_dict=torch.load(opt.weights_path)
        # new_state_dict=OrderedDict()
        # for k,v in state_dict.items():
        #     namekey = k.replace('module_list.', 'module.module_list.')
        #     new_state_dict[namekey] = v
        # model.load_state_dict(new_state_dict)

    model.eval()    # Set in evaluation mode
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    ##############################批处理检测#################################
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        imgs.extend(img_paths)
        img_detections.extend(detections)

    #检测耗时
    current_time = time.time()
    total_time = datetime.timedelta(seconds=current_time - prev_time)
    print(total_time)

    draw_detections_txt(imgs, img_detections, opt.det_folder, opt.img_size)
    # draw_detections(imgs, img_detections, opt.out_folder)

    extract_feat(opt.image_folder, opt.det_folder, opt.roi_folder, opt.csv_folder, opt.mat_folder, opt.start_frame, opt.end_frame, opt.xpart, opt.ypart)
    current_time = time.time()
    #一个timedalta对象代表了一个时间差，当两个date或datetime进行相减操作时会返回一个timedelta对象
    total_time = datetime.timedelta(seconds=current_time - prev_time)
    print(total_time)


if __name__ == "__main__":
    main()
