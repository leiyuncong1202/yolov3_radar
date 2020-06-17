import cv2
import numpy as np
# from sympy import *
import os
import math

def os_cfar(img_name):
    absolute_path='/home/detection/PyTorch-YOLOv3/data/radar1data/grayimages/'+img_name
    img=cv2.imread(absolute_path,0)
    # cv2.imshow("Image",img)
    # cv2.waitKey(0)
    height = img.shape[0]
    width = img.shape[1]
    # print('img_height: %s   img_width: %s' % (height, width))

    #设置参考单元和保护单元的规模，滑窗为ref_size*2+1
    guard_size = 5
    ref_size =round(math.sqrt(6)*guard_size)
    num_guard=pow(guard_size*2+1, 2)-1
    num_ref=pow(ref_size*2+1,2)-num_guard-1
    # print("num_guard: %s   num_ref: %s"%(num_guard,num_ref))
    #设置k值,暂根据经验法则设为3/4N
    k=round(3/4*num_ref)
    #print('k:   %d'%(k))

    #滑窗算子
    tem_thre = np.zeros((height, width))
    x = ref_size*2 +1
    tmp = np.ones((x,x))
    tmp[ref_size-guard_size:ref_size+guard_size+1,ref_size-guard_size:ref_size+guard_size+1] = 0
    for i in range(ref_size, height - ref_size):
        for j in range(ref_size, width - ref_size):
            final = img[i-ref_size:i+ref_size+1,j-ref_size:j+ref_size+1] * tmp
            tem_thre[i][j] = np.sort(final.flatten())[k+(2*guard_size+1)*(2*guard_size+1)]
    #处理外围
    boundary=np.ones((height,width))*70/1.4
    boundary[ref_size:height - ref_size,ref_size:width - ref_size]=0
    final_thre=boundary+tem_thre
    # print('noise_power:   %d' % (ref_amp[k]))

    #设置门限因子为1.4，可根据情况自行调节
    result=np.where(img>final_thre* 1.4,img,0)
    result_path='/home/detection/PyTorch-YOLOv3/data/radar1data/cfar_result/'+ '%s'%(img_name.split('.')[0])+'.png'
    cv2.imwrite(result_path, result)

    return




filePath = '/home/detection/PyTorch-YOLOv3/data/radar1data/grayimages/'
img_names=os.listdir(filePath)
count=0
for img_name in img_names:
    # print(img_name)
    os_cfar(img_name)
    count +=1
    print(count)
