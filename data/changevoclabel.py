# # -*- coding:utf-8 -*-
#
# import xml.etree.ElementTree as ET
# import pickle
# import os
# from os import listdir, getcwd
# from os.path import join
#
# dir='/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/Annotations'
# dir_new='/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/Annotations_boat'
#
# def convert_annotation(name):
#     # 把所有签都修改为boat
#     in_file =open(dir+'/%s.xml'%(name))
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     for obj in root.iter('name'):
#         # print(obj.text)
#         obj.text='boat'
#         # obj.set('updated', 'yes')
#         # obj.find('name').set='ship'
#     tree.write(dir_new+'/%s.xml'%(name))
#
#     # # <filename>改为与XML文件名相同
#     # in_file = open(dir + '/%s.xml' % (name))
#     # tree = ET.parse(in_file)
#     # root = tree.getroot()
#     # new_filename=name + '.' + root.find('filename').text.split('.')[-1]
#     # print(new_filename)
#     # root.find('filename').text=new_filename
#     #
#     # tree.write(dir_new + '/%s.xml' % (name))
#
#
# for file in os.listdir(dir):
#     filename=file.split('.')[0]
#     convert_annotation(filename)


#-*- coding:utf-8 -*-

# 匹配标注和图片
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

dir_xml='/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/Annotations'
dir_img='/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/images'

#缺少标注文件*.xml的图片
xmls=os.listdir(dir_xml)
for file in os.listdir(dir_img):
    filename=file.split('.')[0]+'.xml'
    if filename not in xmls:
        redun_img_path=os.path.join(dir_img,file)
        print(redun_img_path)
        os.remove(redun_img_path)


# #缺少图片文件*.jpg的标注
# imgs=os.listdir(dir_img)
# for file in os.listdir(dir_xml):
#     filename=file.split('.')[0]+'.jpg'
#     if filename not in imgs:
#         redun_xml_path=os.path.join(dir_xml,file)
#         print(redun_xml_path)
#         os.remove(redun_xml_path)

