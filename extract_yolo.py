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
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 PyTorch')
    # 图像与结果文件路径
    #parser.add_argument("--data_folder", type=str, default="data/1114data/", help="path to data") # target2-data 1056*3*4
    #parser.add_argument("--data_folder", type=str, default="data/0315data/", help="path to data") # target3-data 1367*4*4
    parser.add_argument("--data_folder", type=str, default="data/0409data/", help="path to data") # target3-dataR 1367*5*5
    # 配置文件路径
    #parser.add_argument("--data_config", type=str, default="config/1114.data", help="path to data config file") # target2-data
    #parser.add_argument("--data_config", type=str, default="config/0315.data", help="path to data config file") # target3-data
    parser.add_argument("--data_config", type=str, default="config/0409.data", help="path to data config file") # target3-dataR
    parser.add_argument("--model_def", type=str, default="config/yolov3-voc.cfg", help="path to model definition file")
    # 权重文件路径
    #parser.add_argument("--weights_path", type=str, default="weights/yolov3-voc_best.weights", help="path to weights file") # target2-data
    #parser.add_argument("--weights_path", type=str, default="weights/yolov3-voc_0315.pth", help="path to weights file") # target3-data
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-voc_0409.pth", help="path to weights file") # target3-dataR
    parser.add_argument("--class_path", type=str, default="data/ship.names", help="path to class label file")
    # 检测阈值设置：1.conf0.5+nms0.35+globalnms0.35 2.conf0.1+nms0.5+globalnms0.5
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.35, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--global_nms_thres", type=float, default=0.35, help="iou thresshold for global non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension") # 图像尺寸
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument('--start_frame', type=int, default=1, help='start-frame') # 开始帧
    parser.add_argument('--end_frame', type=int, default=1367, help='end-frame') # 结束帧
    # 冗余区域设置：1.416-104 2.608-100 3.1152-128
    parser.add_argument('--x_redundancy', type=int, default=104, help="x-redundancy") # 横向重叠长度
    parser.add_argument('--y_redundancy', type=int, default=104, help="y-redundancy") # 纵向重叠宽度
    # VGG分类权重与特征提取：yolov3_feat/vgg_test.py，当前设置为rect6+fc
    opt = parser.parse_args()
    return opt


# ####################20200509给孙博生成数据文件版本####################
# def draw_detections_txt(imgs, img_detections, det_folder, padded_img_size):
#     if os.path.exists(det_folder):
#         shutil.rmtree(det_folder)
#         os.mkdir(det_folder)
#
#     # for (path, detections) in zip(imgs, img_detections):
#     for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
#         img = np.array(Image.open(path))
#         img_name = path.split('/')[-1].split('.')[0]  # 帧号-列号-行号
#         frame_name = img_name.split('-')[0]  # 图片所属帧号
#         det_result = det_folder + frame_name + ".csv"
#         # print(det_result)
#         imgname = []  # 图片名称
#         bx = []  # ROI左上角的横坐标，基于一帧的大图
#         by = []  # ROI左上角的纵坐标，基于一帧的大图
#         w = []  # ROI宽，基于一帧的大图
#         h = []  # ROI高，基于一帧的大图
#         r = []  # 置信度
#         if detections is not None:
#             detections = rescale_boxes(detections, padded_img_size, img.shape[:2])
#             # print('记录了%s个预测框' % (detections.numpy().shape[0]))
#             for var in (detections.numpy()):
#                 imgname.append(img_name)
#                 bx.append(max(0,var[0]))
#                 by.append(max(0,var[1]))
#                 w.append(max(0,var[2] - var[0]))
#                 h.append(max(0,var[3] - var[1]))
#                 r.append(var[4]*100)
#             df = pd.DataFrame({'imgname': imgname, 'bx': bx, 'by': by, 'w': w, 'h': h, 'r': r})
#             df.to_csv(det_result, sep=' ', mode='a', header=False, index=0)
#         else:
#             if not os.path.exists(det_result):
#                 os.mknod(det_result)


# ####################20200521给孙博生成数据文件版本####################
# def global_draw_detections_txt(global_imgs, global_detections, global_det_folder):
#     if os.path.exists(global_det_folder):
#         shutil.rmtree(global_det_folder)
#         os.mkdir(global_det_folder)
#     print(global_imgs)
#     for img_i, (path, detections) in enumerate(zip(global_imgs, global_detections)):
#         img_name = path.split('/')[-1].split('.')[0]  # 帧号-列号-行号
#         frame_name = img_name.split('-')[0]  # 图片所属帧号
#         print(img_name,frame_name)
#         det_result = global_det_folder + frame_name + ".csv"
#         # print(det_result)
#         imgname = []  # 图片名称
#         x = []  # 同bx, 左上角的横坐标
#         y = []  # 同by, 左上角的纵坐标
#         w = []  # ROI宽，基于一帧的大图
#         h = []  # ROI高，基于一帧的大图
#         r = []  # 置信度
#         if detections is not None:
#             # print('记录了%s个预测框' % (detections.numpy().shape[0]))
#             for var in (detections.numpy()):
#                 imgname.append(img_name)
#                 x.append(max(0, var[0]))
#                 y.append(max(0, var[1]))
#                 w.append(max(0, var[2] - var[0]))
#                 h.append(max(0, var[3] - var[1]))
#                 r.append(var[4])
#             df = pd.DataFrame(
#                 {'imgname': imgname, 'x': x, 'y': y, 'w': w, 'h': h, 'r': r})
#             df.to_csv(det_result, sep=' ', mode='a', header=False, index=0)
#         else:
#             if not os.path.exists(det_result):
#                 os.mknod(det_result)


def global_draw_detections_txt(global_imgs, global_detections, global_det_folder):
    if os.path.exists(global_det_folder):
        shutil.rmtree(global_det_folder)
        os.mkdir(global_det_folder)

    for img_i, (path, detections) in enumerate(zip(global_imgs, global_detections)):
        img_name = path.split('/')[-1].split('.')[0]  # 帧号-列号-行号
        frame_name = img_name.split('-')[0]  # 图片所属帧号
        det_result = global_det_folder + frame_name + ".csv"
        # print(det_result)
        imgname = []  # 图片名称
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
                bx.append(max(0, var[0]))
                by.append(max(0, var[1]))
                x.append(max(0, var[0]))
                y.append(max(0, var[1]))
                w.append(max(0, var[2] - var[0]))
                h.append(max(0, var[3] - var[1]))
                r.append(var[4])
                fr.append(frame_name)
            df = pd.DataFrame(
                {'imgname': imgname, 'bx': bx, 'by': by, 'x': x, 'y': y, 'w': w, 'h': h, 'r': r, 'fr': fr})
            df.to_csv(det_result, sep=' ', mode='a', header=False, index=0)
        else:
            if not os.path.exists(det_result):
                os.mknod(det_result)


def draw_detections_txt(imgs, img_detections, det_folder, padded_img_size):
    if os.path.exists(det_folder):
        shutil.rmtree(det_folder)
        os.mkdir(det_folder)

    # for (path, detections) in zip(imgs, img_detections):
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        img = np.array(Image.open(path))
        img_name = path.split('/')[-1].split('.')[0]  # 帧号-列号-行号
        frame_name = img_name.split('-')[0]  # 图片所属帧号
        det_result = det_folder + frame_name + ".csv"
        # print(det_result)
        imgname = []  # 图片名称
        bx = []  # ROI左上角的横坐标,基于一帧的大图
        by = []  # ROI左上角的纵坐标，基于一帧的大图
        x = []  # 同bx, 左上角的横坐标
        y = []  # 同by, 左上角的纵坐标
        w = []  # ROI宽，基于一帧的大图
        h = []  # ROI高，基于一帧的大图
        r = []  # 置信度
        fr = []  # 帧数
        if detections is not None:
            # detections = rescale_boxes(detections, padded_img_size, img.shape[:2])
            # print('记录了%s个预测框' % (detections.numpy().shape[0]))
            for var in (detections.numpy()):
                imgname.append(img_name)
                bx.append(max(0,var[0]))
                by.append(max(0,var[1]))
                x.append(max(0,var[0]))
                y.append(max(0,var[1]))
                w.append(max(0,var[2] - var[0]))
                h.append(max(0,var[3] - var[1]))
                r.append(var[4]*100)
                fr.append(frame_name)
            df = pd.DataFrame({'imgname': imgname, 'bx': bx, 'by': by, 'x': x, 'y': y, 'w': w, 'h': h, 'r': r, 'fr': fr})
            df.to_csv(det_result, sep=' ', mode='a', header=False, index=0)
        else:
            if not os.path.exists(det_result):
                os.mknod(det_result)


#####################在全局图片上绘制检测框并保存####################
def global_draw_detections(global_imgs, global_detections, out_folder,s_img_detections):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, g_detections,s_detections) in enumerate(zip(global_imgs, global_detections,s_img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))
        filename = out_folder + path.split("/")[-1].split(".")[0] + '.png'
        print(filename + '**************************')
        global_image=cv2.imread(path)
        if s_detections is not None:
            for x1, y1, x2, y2, conf, fr in s_detections:
                global_image = cv2.rectangle(global_image, (x1, y2), (x2, y1), (255, 0, 0), 2)
        if g_detections is not None:
            for x1, y1, x2, y2, conf, fr in g_detections:
                global_image=cv2.rectangle(global_image,(x1,y2),(x2,y1),(0,0,255),2)
                cv2.putText(global_image,'{:.2f}'.format(conf),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        cv2.imwrite(filename,global_image)


####################在图片上绘制检测框并保存####################
def draw_detections(imgs, img_detections, out_folder, padded_img_size, classes):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)

    # Bounding-box colors，使用matplotlib量化离散化的色图
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

        # 绘制GT框
        # print(len(targets))
        # if gts is not None:
        #     for gt_index,class_num,gt_x,gt_y,gt_w,gt_h in gts:
        #         gt_x *= img.shape[1]
        #         gt_y *= img.shape[0]
        #         gt_w *= img.shape[1]
        #         gt_h *= img.shape[0]
        #         print(gt_x,gt_y,gt_w,gt_h)
        #         box_gt = patches.Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=2, edgecolor="blue", facecolor="none")
        #         ax.add_patch(box_gt)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            # img.shape[:2]输出前两维，即原始高度和宽度
            detections = rescale_boxes(detections, padded_img_size, img.shape[:2])
            # 一张图上有多个框时，从colors随机提取出n_cls_preds个元素
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred, level_index in detections:  # 层级索引
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                # 获取一个位置的值，并将其作为python的数据类型
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="red", facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label图片上的标注
                plt.text(
                    x1,
                    y1,
                    s='{:.2f}'.format(conf),
                    color="yellow",
                    verticalalignment="top",
                    bbox={"ec": "black", "pad": 0},
                )
        # Save generated image with detections
        # 不显示坐标尺寸
        plt.axis("off")
        # 删除坐标轴的刻度显示
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        print(out_folder + f"{filename}.png"+'**************************')
        # 保存pad为0的紧框，即只有该图片
        plt.savefig(out_folder + f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()


def main():
    opt = parse_args()
    print(opt)
    image_folder=opt.data_folder + 'images/'  # 原始图片的位置
    det_folder=opt.data_folder + 'detResult/'  # 检测结果的文件位置，每帧的所有小图的检测结果存放于一个文件夹下
    out_folder=opt.data_folder + 'output/'  # 检测结果的图片形式输出
    roi_folder=opt.data_folder + 'ROI/'  # 裁剪的ROI图片的输出
    csv_folder=opt.data_folder + 'csv/'  # csv格式的特征输出
    mat_folder=opt.data_folder + 'mat/'  # mat格式的特征输出
    global_img_folder=opt.data_folder + 'global_images/'  # 原始大图的位置
    global_det_folder=opt.data_folder + 'g_detResult/'  # 全局检测结果的文件位置
    prev_time = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    # ####################预测图片与预测结果的存放位置####################
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(det_folder):
        os.mkdir(det_folder)
    if not os.path.exists(roi_folder):
        os.mkdir(roi_folder)
    if not os.path.exists(global_det_folder):
        os.mkdir(global_det_folder)
    print("\n")
    print("*" * 50)
    print("开始检测")
    print("*" * 50)
    ####################加载数据####################
    # Get dataloader
    # ####################带有标注的dataloader####################
    # data_config = parse_data_config(opt.data_config)
    # path = data_config["valid"]
    # dataset = ListDataset(path, img_size=opt.img_size, augment=False, multiscale=False)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn
    # )

    ####################不带标注的dataloader####################
    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    ####################加载模型和权重####################
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # 单机多卡
    # device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    # model = Darknet(opt.model_def, img_size=opt.img_size)
    # model = nn.DataParallel(model).to(device)

    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        # 单卡
        model.load_state_dict(torch.load(opt.weights_path))
        # 多卡
        # state_dict=torch.load(opt.weights_path)
        # new_state_dict=OrderedDict()
        # for k,v in state_dict.items():
        #     namekey = k.replace('module_list.', 'module.module_list.')
        #     new_state_dict[namekey] = v
        # model.load_state_dict(new_state_dict)

    model.eval()  # Set in evaluation mode
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    targets = []
    ####################带有标注的批处理检测####################
    # for batch_i, (img_paths, input_imgs, gts) in enumerate(dataloader):
    #     input_imgs = Variable(input_imgs.type(Tensor))
    #     with torch.no_grad():
    #         detections = model(input_imgs)
    #         detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    #     imgs.extend(img_paths)
    #     img_detections.extend(detections)
    #     targets.extend(gts)
    ####################没有标注的批处理检测####################
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # 检测耗时
    # current_time = time.time()
    # total_time = datetime.timedelta(seconds=current_time - prev_time)
    # print(total_time)

    # ####################绘制检测结果
    # draw_detections(imgs, img_detections, out_folder, opt.img_size, classes)

    # ####################普通NMS
    # draw_detections_txt(imgs, img_detections, det_folder, opt.img_size)
    # ####################全局NMS
    draw_detections_txt(imgs, img_detections, det_folder, opt.img_size)
    global_img_detections = []  # Stores detections for global image
    #s_img_detections = []  # 未经过全局NMS的检测结果
    global_imgs=[]
    filenames = os.listdir(det_folder)
    print(len(filenames))
    for i in range(1, len(filenames)+1):
        global_outputs=readcsv(det_folder + str(i)+'.csv',opt.img_size,opt.x_redundancy,opt.y_redundancy)
        #s_img_detections.append(global_outputs)
        global_outputs=global_nms(global_outputs,nms_thres=opt.global_nms_thres)
        # print("global_outputs: ",global_outputs)
        global_img_detections.append(global_outputs)
        # print(len(global_outputs))
        global_img_path=global_img_folder+str(i)+'.png'
        global_imgs.append(global_img_path)
    # print("img_detections[0]: ",img_detections[0])
    # print("global_img_detections[0]: ",global_img_detections[0])
    global_draw_detections_txt(global_imgs, global_img_detections, global_det_folder)
    # 绘制全局大图的检测结果
    #global_draw_detections(global_imgs, global_img_detections, out_folder,s_img_detections)

    # ####################提取特征
    extract_feat(global_img_folder, global_det_folder, roi_folder, csv_folder, mat_folder, opt.start_frame, opt.end_frame,opt.x_redundancy,opt.y_redundancy)
    current_time = time.time()
    # 一个timedalta对象代表了一个时间差，当两个date或datetime进行相减操作时会返回一个timedelta对象
    total_time = datetime.timedelta(seconds=current_time - prev_time)
    print(total_time)


if __name__ == "__main__":
    main()
