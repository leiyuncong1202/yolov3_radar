from __future__ import division
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import csv


##########202004版本，在大图上裁剪：将ROI上下左右皆扩展5个像素来提取；然后，填充黑边至112*112
def cropImage(filename, rootpath, ROIpath):
    imgname=[]
    left = []
    top = []
    width=[]
    height=[]
    score=[]
    frame_name = []

    imgname_new=[]
    left_new=[]
    top_new=[]
    width_new=[]
    height_new=[]
    score_new=[]
    frame_new=[]

    f=open(filename)
    lines=f.readlines()
    for line in lines:
        imgname.append(line.split()[0])
        left.append(float(line.split()[1]))
        top.append(float(line.split()[2]))
        width.append(float(line.split()[5]))
        height.append(float(line.split()[6]))
        score.append(float(line.split()[7]))
        frame_name.append(line.split()[8])
    f.close()

    # 把图片按名字排序
    imgnamet, post, countt = np.unique(imgname, return_counts=True, return_index=True)

    # 裁剪ROI图片并保存
    for (imageName,pos,count) in zip(imgnamet, post, countt):
        img = cv2.imread(rootpath + imageName + '.png')
        for i in range(pos,pos+count):
            # print(filename)
            # print(i)
            # 上下左右各扩充5个像素后裁剪
            crop_img_xmin=max(0,round(left[i]-5))
            crop_img_xmax=min(img.shape[1],round(left[i] + width[i] + 5))
            crop_img_ymin=max(0,round(top[i]-5))
            crop_img_ymax=min(img.shape[0],round(top[i] + height[i] + 5))
            crop_img=img[crop_img_ymin:crop_img_ymax,crop_img_xmin:crop_img_xmax]
            # 填充0至112*112
            top_border=int((112-crop_img.shape[0])/2)
            bottom_border=112-top_border-crop_img.shape[0]
            left_border=int((112-crop_img.shape[1])/2)
            right_border=112-left_border-crop_img.shape[1]

            print(imageName,top_border,left_border)
            if(top_border>0 and right_border>0):
                crop_img=cv2.copyMakeBorder(crop_img,top_border,bottom_border,left_border,right_border,cv2.BORDER_CONSTANT,0)
                # 裁剪
                ROIName=ROIpath + imageName+ "-%s.png"%(str(i-pos))
                cv2.imwrite(ROIName, crop_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

                imgname_new.append(imageName+ "-%s"%(str(i-pos)))
                left_new.append(left[i])
                top_new.append(top[i])
                width_new.append(width[i])
                height_new.append(height[i])
                score_new.append(score[i])
                frame_new.append(frame_name[i])

    df = pd.DataFrame({'imgname': imgname_new, 'bx': left_new, 'by': top_new, 'x': left_new, 'y': top_new, 'w': width_new, 'h': height_new, 'r': score_new, 'fr': frame_new})
    df.to_csv(filename, sep=' ', mode='w', header=False, index=0)

# 根据txt文件和png图片，裁剪ROI
def crop(rootpath, detpath, ROIpath, start_frame, end_frame):
    print("\n")
    print("*" * 50)
    print("开始裁剪ROI")
    print("*" * 50)

    if os.path.exists(ROIpath):
        shutil.rmtree(ROIpath)
        os.mkdir(ROIpath)
    # 按帧裁剪
    for frame in range(start_frame, end_frame+1):
        fileName = detpath + str(frame) + '.csv' # 大图对应的检测结果
        # print(fileName)
        cropImage(fileName, rootpath, ROIpath)


# ##########2019初始版本，在小图上按照ROI的准确尺寸进行裁剪
# def cropImage(filename, rootpath, ROIpath):
#     imgname=[]
#     left = []
#     top = []
#     width=[]
#     height=[]
#     score=[]
#     frame_name = []
#
#     imgname_new=[]
#     left_new=[]
#     top_new=[]
#     width_new=[]
#     height_new=[]
#     score_new=[]
#
#     # 读入TXT文件
#     # f = open(filename)
#     # line = f.readline()
#     # while line:
#     #     imgname.append(line.split()[0])
#     #     left.append(float(line.split()[1]))
#     #     top.append(float(line.split()[2]))
#     #     width.append(float(line.split()[5]))
#     #     height.append(float(line.split()[6]))
#     #     score.append(float(line.split()[7]))
#     #     frame_name.append(line.split()[8])
#     #     line = f.readline()
#     # f.close()
# # readlines比readline效率高，内存占用大
#
#     f=open(filename)
#     lines=f.readlines()
#     for line in lines:
#         imgname.append(line.split()[0])
#         left.append(float(line.split()[1]))
#         top.append(float(line.split()[2]))
#         width.append(float(line.split()[5]))
#         height.append(float(line.split()[6]))
#         score.append(float(line.split()[7]))
#         frame_name.append(line.split()[8])
#     f.close()
#
#     # 把图片按名字排序
#     imgnamet, post, countt = np.unique(imgname, return_counts=True, return_index=True)
#
#     # 裁剪ROI图片并保存
#     for (imageName,pos,count) in zip(imgnamet, post, countt):
#         img = cv2.imread(rootpath + imageName + '.png')
#         for i in range(pos,pos+count):
#             # print(filename)
#             # print(i)
#             crop_img = img[round(top[i]):round(top[i] + height[i] + 0.5),
#                        round(left[i]):round(left[i] + width[i] + 0.5)]
#             ROIName=ROIpath + imageName+ "-%s.png"%(str(i-pos))
#             imgname_new.append(imageName+ "-%s"%(str(i-pos)))
#             left_new.append(left[i])
#             top_new.append(top[i])
#             width_new.append(width[i])
#             height_new.append(height[i])
#             score_new.append(score[i])
#             # print(ROIName)
#             cv2.imwrite(ROIName, crop_img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
#
#     df = pd.DataFrame({'imgname': imgname_new, 'bx': left_new, 'by': top_new, 'x': left_new, 'y': top_new, 'w': width_new, 'h': height_new, 'r': score_new, 'fr': frame_name})
#     df.to_csv(filename, sep=' ', mode='w', header=False, index=0)
#
# # 根据txt文件和png图片，裁剪ROI
# def crop(rootpath, detpath, ROIpath, start_frame, end_frame):
#     print("\n")
#     print("*" * 50)
#     print("开始裁剪ROI")
#     print("*" * 50)
#
#     if os.path.exists(ROIpath):
#         shutil.rmtree(ROIpath)
#         os.mkdir(ROIpath)
#     # 按帧裁剪
#     for frame in range(start_frame, end_frame+1):
#         fileName = detpath + str(frame) + '.csv' # 小图对应的检测结果
#         # print(fileName)
#         cropImage(fileName, rootpath, ROIpath)
