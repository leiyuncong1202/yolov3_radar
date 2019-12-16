# -*- coding:utf-8 -*- #
from yolov3_feat.crop import crop
from yolov3_feat.vgg_test import vgg_test
from yolov3_feat.concat import concat
import pandas as pd
import os


def extract_feat(image_folder, det_folder, roi_folder, csv_folder, mat_folder, start_frame, end_frame, xpart, ypart):
    ROIfile = roi_folder + 'ROI_feat.csv'  # 存储CNN特征的文件

    # 根据检测文件和416*416的图片，裁剪ROI
    crop(image_folder, det_folder, roi_folder, start_frame, end_frame, xpart, ypart)
    # 将ROI送入VGG网络提取特征
    vgg_test(roi_folder, ROIfile)
    # 转换至基于大图的坐标，合并坐标、特征等信息，保存至mat文件或csv文件
    concat(det_folder, ROIfile, csv_folder, mat_folder)


