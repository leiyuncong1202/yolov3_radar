# 有标注文件时
import os

fvalid = open('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/testfilename.txt', 'w+')
i =0
c_test=0
for img in os.listdir('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/images'):
    # print(img)
    if img.split('.')[-1]=='jpg':
        imgpath = os.path.join('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/images', img)
        labelpath = os.path.join('/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/labels', img.split('.')[0] + '.txt')
        # 删除无GT的图片
        label_size = os.path.getsize(labelpath)
        if label_size==0:
            i += 1
            continue
            # print(i)
        else:
            fvalid.writelines(imgpath)
            fvalid.write('\n')
            c_test+=1
fvalid.close()
print("无GT框的图片：%s张"%i)
print("测试集：%s张"%c_test)


# # 无标注文件时
# # import os
# #
# # fvalid = open('/home/detection/projects/PyTorch-YOLOv3/data/0521data/416/testfilename.txt', 'w+')
# # for img in os.listdir('/home/detection/projects/PyTorch-YOLOv3/data/0521data/416/images'):
# #     print(img)
# #     if img.split('.')[-1]=='png':
# #         imgpath = os.path.join('/home/detection/projects/PyTorch-YOLOv3/data/0521data/416/images', img)
# #         fvalid.writelines(imgpath)
# #         fvalid.write('\n')
# # fvalid.close()