import os
import numpy as np
import random

# 比例
validation_percentage=10

# 跳过无GT框的图片；划分训练集—验证集
def split_train_valid():
    v = 0
    t = 0
    zero_count=0
    fvalid = open('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/valid.txt', 'w+')
    ftrain = open('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/train.txt', 'w+')

    for img in os.listdir('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/images'):
        imgpath=os.path.join('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/images',img)
        # 跳过无GT框的图片
        labelpath = os.path.join('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/labels', img.split('.')[0]+'.txt')
        label_size = os.path.getsize(labelpath)
        if label_size==0:
            # print(labelpath)
            zero_count += 1
            continue
        chance=np.random.randint(100)
        if chance<validation_percentage:
            fvalid.writelines(imgpath)
            fvalid.write('\n')
            v += 1
        else:
            ftrain.writelines(imgpath)
            ftrain.write('\n')
            t += 1

    ftrain.close()
    fvalid.close()
    print("无GT框的图片：%s张"%zero_count)
    print("训练集：%s张；验证集：%s张"%(t,v))



# def del_null_image():
#     i =0
#     for lab in os.listdir('/home/detection/PyTorch-YOLOv3/data/radar3data/labels'):
#         labelpath=os.path.join('/home/detection/PyTorch-YOLOv3/data/radar3data/labels',lab)
#         label_size = os.path.getsize(labelpath)
#         if label_size==0:
#             print(labelpath)
#             i += 1
#             continue
#     print(i)


def main():
    split_train_valid()

if __name__=='__main__':
    main()