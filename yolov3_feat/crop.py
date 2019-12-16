from __future__ import division
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import csv

def cropImage(filename, rootpath, ROIpath):
    imgname=[]
    left = []
    top = []
    width=[]
    height=[]
    score=[]
    frame_name = []
    imgname_new=[]

    # 读入TXT文件
    # f = open(filename)
    # line = f.readline()
    # while line:
    #     imgname.append(line.split()[0])
    #     left.append(float(line.split()[1]))
    #     top.append(float(line.split()[2]))
    #     width.append(float(line.split()[5]))
    #     height.append(float(line.split()[6]))
    #     score.append(float(line.split()[7]))
    #     frame_name.append(line.split()[8])
    #     line = f.readline()
    # f.close()
###readlines 比readline 效率高  内存占用大

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



    imgnamet, post, countt = np.unique(imgname, return_counts=True, return_index=True)

    # 裁剪ROI图片并保存
    for (imageName,pos,count) in zip(imgnamet, post, countt):
        img = cv2.imread(rootpath + imageName + '.png')
        for i in range(pos,pos+count):
            # print(filename)
            # print(i)
            crop_img = img[round(top[i]):round(top[i] + height[i] + 0.5),
                       round(left[i]):round(left[i] + width[i] + 0.5)]
            ROIName=ROIpath + imageName+ "-%s.png"%(str(i-pos))
            imgname_new.append(imageName+ "-%s"%(str(i-pos)))
            # print(ROIName)
            cv2.imwrite(ROIName, crop_img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

    df = pd.DataFrame({'imgname': imgname_new, 'bx': left, 'by': top, 'x': left, 'y': top, 'w': width, 'h': height, 'r': score, 'fr': frame_name})
    df.to_csv(filename, sep=' ', mode='w', header=False, index=0)







# 根据txt文件和png图片，裁剪ROI
def crop(rootpath, detpath, ROIpath, start_frame, end_frame, xpart, ypart):
    print("\n")
    print("*" * 50)
    print("开始裁剪ROI")
    print("*" * 50)

    if os.path.exists(ROIpath):
        shutil.rmtree(ROIpath)
        os.mkdir(ROIpath)
    # 按帧裁剪
    for frame in range(start_frame, end_frame+1):
        fileName = detpath + str(frame) + '.csv'  #小图对应的检测结果
        # print(fileName)
        cropImage(fileName, rootpath, ROIpath)



# if __name__ == '__main__':
#     main()
