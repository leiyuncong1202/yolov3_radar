import os
import sys
import xml.etree.ElementTree as ET
import glob

# 图片尺寸
width = 416
height =416

def xml_to_txt(indir,outdir):

    # os.chdir(indir)
    # annotations = os.listdir('.')
    # print(annotations)
    annotations = glob.glob(indir+'/*.xml')

    for i, file in enumerate(annotations):

        file_save = file.split('.')[0].split('/')[-1]+'.txt'
        # print(file_save)
        file_txt=os.path.join(outdir,file_save)
        print(file_txt)
        f_w = open(file_txt,'w')

        # actual parsing
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
                current = list()
                # name = obj.find('name').text

                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)

                label_idx_x='0'
                x_center=((xmin+xmax)/2)/width
                y_center=((ymin+ymax)/2)/height
                w = (xmax-xmin)/width
                h = (ymax-ymin)/height
                f_w.write(label_idx_x+' '+str(x_center)+' '+str(y_center)+' '+str(w)+' '+str(h)+'\n')
                

indir='/home/detection/PyTorch-YOLOv3/data/1114data/Annotations'   #xml目录
outdir='/home/detection/PyTorch-YOLOv3/data/1114data/labels'  #txt目录

xml_to_txt(indir,outdir)
