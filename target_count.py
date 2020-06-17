import cv2
import numpy as np
import os

def count_target(img_name):
    #打开图片
    # img=cv2.imread('D:\\Python\\CFAR\\target1-result\\1-0.png',0)
    absolute_path = '/home/detection/PyTorch-YOLOv3/data/radar1data/cfar_result/' + img_name
    img = cv2.imread(absolute_path, 0)

    #构造模板，3次腐蚀，3次膨胀，得到背景
    kernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(img,kernel,iterations=3)
    dilation=cv2.dilate(erosion,kernel,iterations=3)

    #原图减去背景得到米粒形状
    backImg=dilation
    target_con=img-backImg

    #OSTU二值化
    th1,ret1=cv2.threshold(target_con,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #轮廓检测
    contours,hierarchy=cv2.findContours(ret1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    #遍历得到最大面积的米粒
    maxS=20
    minS=10
    tmp=[]
    for cnt in contours:
        tempS=cv2.contourArea(cnt)
        if maxS>tempS>minS:
            tmp.append(cnt)


    #在img中画出最大面积米粒
    cv2.drawContours(img,tmp,-1,(0,0,255,),3)
    contour_path='/home/detection/PyTorch-YOLOv3/data/radar1data/contour/'+ '%s'%(img_name.split('.')[0])+'.png'
    cv2.imwrite(contour_path,target_con)
    return len(tmp)


if __name__=='__main__':
    filePath = '/home/detection/PyTorch-YOLOv3/data/radar1data/cfar_result/'
    img_names=os.listdir(filePath)
    count=0
    for img_name in img_names:
        # print(img_name)
        count_target(img_name)
        count +=1
        print(count)


#     # print('目标个数：',len(tmp))
#     # cv2.imshow('image',target_con)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
# filename = '/home/detection/PyTorch-YOLOv3/data/radar1data/write_data.txt'
# target_num=[]
# failed_image=[]
# with open(filename) as f:
#     while 1:
#         lines = f.readlines(3353)
#         if not lines:
#             break
#         for line in lines:
#             # print(line.split('\\')[0])
#             s=line.split()
#             # print(s)
#             # print(s[1])
#             target_num.append(int(s[1]))
#             if int(s[1])!=10:
#                 failed_image.append(s)
#                 print(s)
# print('平均值： ',np.mean(target_num))
# print('失败图片数： ',len(failed_image))
# f.close()

