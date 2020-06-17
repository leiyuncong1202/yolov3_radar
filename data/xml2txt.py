import os
import sys
import xml.etree.ElementTree as ET
import glob

# # 图片尺寸
# width = 416
# height =416

def xml_to_txt(indir,outdir,clsText):
    clsDict = {}
    i=0
    with open(clsText) as f:
        fLines=f.readlines()
    for fl in fLines:
        clsDict[fl.strip()]=i
        i+=1
    # print(clsDict)          # 打印类别名称


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

        img_size=root.find('size')
        width=float(img_size.find('width').text)
        height=float(img_size.find('height').text)
        # 多分类时，从xml读取类名
        for obj in root.iter('object'):
                current = list()
                clsName = obj.find('name').text
                if(clsName in clsDict):
                    label_idx_x=str(clsDict[clsName])
        # 二分类时，直接指定类名
        # label_idx_x='0'

                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)

                x_center=((xmin+xmax)/2)/width
                y_center=((ymin+ymax)/2)/height
                w = (xmax-xmin)/width
                h = (ymax-ymin)/height
                f_w.write(label_idx_x+' '+str(x_center)+' '+str(y_center)+' '+str(w)+' '+str(h)+'\n')
                

indir='/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/Annotations'   #xml目录
outdir='/home/detection/projects/PyTorch-YOLOv3/data/0601_cv_ship/labels'  #txt目录
clsText='/home/detection/projects/PyTorch-YOLOv3/data/multi_ship.names'

xml_to_txt(indir,outdir,clsText)
