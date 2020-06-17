# -*- coding:utf-8 -*- #
import os
import os.path
import time
import numpy as np
import pandas as pd
import csv
import scipy.io as io
import shutil
import multiprocessing
import math

img_width=416
img_height=416


# 将cnn特征向量添加至csv或mat中
def addFeature(outfile,ROIfile,resultfile,matfile,x_redundancy, y_redundancy):
    imgname = []  # ROI图片的名字，a-b-c-d: a帧数，b列，c行，d所在小图上的ROI索引
    bx = []  # ROI左上角的横坐标,基于一帧的大图
    by = []  # ROI左上角的纵坐标，基于一帧的大图
    x = []  # 同bx, 左上角的横坐标
    y = []  # 同by, 左上角的纵坐标
    w = []  # ROI宽，基于一帧的大图
    h = []  # ROI高，基于一帧的大图
    r = []  # 置信度
    fr = []  # 帧数
    imgname2 = []  # cnn文件中存储的名字，与imgname对应
    cnn_feat = []  # 由ROI图片提取的256维向量
    column_num = []
    row_num = []

    f = open(outfile)
    lines = f.readlines()
    for line in lines:
        imgname.append(line.split()[0])
        ################转换为基于大图的坐标########################
        # print(int(line.split()[0].split('-')[0]),int(line.split()[0].split('-')[1]), img_width, float(line.split()[1]))
        bx.append([float(line.split()[1])])
        by.append([float(line.split()[2])])
        w.append([float(line.split()[5])])
        h.append([float(line.split()[6])])
        r.append([float(line.split()[7])])
        fr.append([float(line.split()[8])])
        # column_num.append(int(line.split()[0].split('-')[1]))
        # row_num.append(int(line.split()[0].split('-')[2]))
    f.close()

    x = bx
    y = by
    df1 = pd.DataFrame(
        {'imgname': imgname, 'bx': bx, 'by': by, 'x': x, 'y': y, 'w': w, 'h': h, 'r': r, 'fr': fr})

    with open(ROIfile,'r' ) as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            imgname2.append(line[0][2:-5])
            line[1] = line[1].split('[')[-1]
            line[256] = line[256].split(')')[0].split(']')[0]
            cnntensor = line[1:257]
            float_cnntensor = [float(x) for x in cnntensor]
            # print(float_cnntensor)
            cnn_feat.append(float_cnntensor)
            # print(len(cnn_feat))

    df2=pd.DataFrame({'imgname':imgname2, 'cnn_feat':cnn_feat})
    # 以'imgname'为key进行左连接合并
    result=df1.join(df2.set_index('imgname'), on='imgname',how='inner')

    # print("此帧共有%s个ROI"%result.shape[0])
    # 存储至csv格式
    # result.to_csv(resultfile, sep=',', mode='w', header=False, index=0)
    # 存储至mat格式
    if not os.path.exists(matfile):
        os.mknod(matfile)
    print(matfile)
    io.savemat(matfile, {'det':{'bx': list(result['bx']),
                 'by': list(result['by']),
                 'x' : list(result['x']),
                 'y' : list(result['y']),
                 'w' : list(result['w']),
                 'h' : list(result['h']),
                 'r' : list(result['r']),
                 'fr' : list(result['fr']),
                 'cnn': list(result['cnn_feat'])
                 }})



def concat(detpath, ROIfile, csvpath, matpath, x_redundancy, y_redundancy):
    if os.path.exists(csvpath):
        shutil.rmtree(csvpath)
    os.mkdir(csvpath)
    if os.path.exists(matpath):
        shutil.rmtree(matpath)
    os.mkdir(matpath)
    print("\n")
    print("*" * 50)
    print("开始合并检测文件和特征文件")
    print("*" * 50)

    m = 8
    filenames=os.listdir(detpath)
    print(len(filenames))
    n = int(math.ceil(len(filenames) / float(m)))
    pool = multiprocessing.Pool(processes=m)
    for i in range(0, len(filenames), n):
        pool.apply_async(mutliprocess, (filenames[i:i + n],detpath,csvpath,matpath,ROIfile,x_redundancy,y_redundancy))

    pool.close()
    pool.join()



def mutliprocess(msg,detpath,csvpath,matpath,ROIfile,x_redundancy,y_redundancy):
    # print(len(msg))
    for i in msg:
        # print("开始合并第%s帧图片" % i)  # 第几帧图片
        outfile = detpath + str(i)
        csvfile = csvpath + str(i).split('.')[0] + '.csv'  # 最终csv形式
        matfile = matpath + str(i).split('.')[0] + '.mat'  # 最终mat形式
        addFeature(outfile, ROIfile, csvfile, matfile,x_redundancy,y_redundancy)



    # for i in os.listdir(detpath):
    #     # if os.path.isdir(detpath+str(i)):
    #     print("开始合并第%s帧图片"%i)     # 第几帧图片
    #     # dirpath=detpath+str(i)
    #     outfile=detpath+str(i)
    #     csvfile=csvpath+str(i).split('.')[0]+'.csv'  # 最终csv形式
    #     matfile=matpath+str(i).split('.')[0]+'.mat'        # 最终mat形式
    #     # mergeTxt(dirpath,outfile)         # 合并一帧上的所有小图，并将坐标转换为基于大图的值
    #     addFeature(outfile,ROIfile,csvfile,matfile)



# if __name__ == '__main__':
#     main()


