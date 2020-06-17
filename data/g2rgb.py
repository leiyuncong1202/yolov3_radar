import cv2
import os
import numpy as np


if __name__=='__main__':
    dir_imggray='/home/detection/PyTorch-YOLOv3/data/radar1data/grayimages'
    dir_imgcolor='/home/detection/PyTorch-YOLOv3/data/radar1data/images'
    # color_shape = cv2.imread('/home/detection/PyTorch-YOLOv3/data/radar3data/images/137-1-2.png')
    count=0

    files=os.listdir(dir_imggray)
    for file in files:
        path_imggray=dir_imggray+'/'+file
        path_imgcolor=dir_imgcolor+'/'+file
        imggray = cv2.imread(path_imggray,cv2.IMREAD_GRAYSCALE)
        imgcolor=np.zeros((416,416,3))
        # 复制灰度图片至伪彩色图片的每一个通道
        imgcolor[:, :, 0] = imggray
        imgcolor[:, :, 1] = imggray
        imgcolor[:, :, 2] = imggray
        cv2.imwrite(path_imgcolor,imgcolor)

        count += 1
        print(count)
        print(imggray.shape)
        print(imgcolor.shape)