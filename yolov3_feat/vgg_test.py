# coding: utf-8
from __future__ import print_function, division

import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torch.utils.data as Data
from torchvision import datasets, transforms
from torchvision.datasets.folder import DatasetFolder
from torch.autograd import Variable
import numpy as np
from torchvision import models
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA



#####################指定batch_size#############################
batch_size = 32
num_workers = 32
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 图片resize等预处理
test_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


# VGG网络定义
class VGGNet(nn.Module):
    def __init__(self, num_classes=11):  # num_classes，
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)  # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 4096),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 256),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 自定义数据集，便于将cnnfeature与图片名字对应
class our_datasets(Data.Dataset):
    def __init__(self, root,batch_size=1,transform=None):
        self.root = root
        imgs_list =os.listdir(root)
        self.imgs_list = [os.path.join(root,i) for i in imgs_list]
        self.batch_size = batch_size
        self.transform = transform

    def __getitem__(self, index ):
        img_path = self.imgs_list[index]
        # print(img_path)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img_data = self.transform(img)
        label = img_path.split('/')[-1]
        return img_data, label

    def __len__(self):
        return len(self.imgs_list)





# 测试
def inference(test_dir):
    test_dataset = our_datasets(test_dir, transform=test_transforms)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    feature_tensor = []
    feature_frame = []
    model = VGGNet()
    # print(model)

    # 加载分类层之前的所有权重
    state_dict = torch.load(r'/home/detection/PyTorch-YOLOv3/yolov3_feat/29vgg16classifier.pth')['state']
    del state_dict['classifier.3.weight']
    del state_dict['classifier.3.bias']
    del state_dict['classifier.6.weight']
    del state_dict['classifier.6.bias']
    # for key, param in state_dict.items():
    #     print(key)
    model.load_state_dict(state_dict)
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 测试模式
    model.eval()
    # 批测试
    for batch_x,imgname in test_dataloader:
        batch_x=Variable(batch_x,requires_grad=False).cuda()
        # print(batch_x,imgname)
        with torch.no_grad():
            result=model(batch_x)
            # print(result,imgname)
        feature_frame.extend(imgname)
        feature_tensor.extend(result.cpu().numpy().tolist())

    print("所有图片共生成%s个ROI"%(len(feature_frame)))
    return (feature_frame,feature_tensor)



def vgg_test(test_dir, filename):
    print("\n")
    print("*" * 50)
    print("开始提取ROI特征")
    print("*" * 50)

    feature_frame,feature_tensor = inference(test_dir)

    np_cnn_feat=np.asarray(feature_tensor)
    # print(np_cnn_feat.shape)
    pca = PCA(n_components=256,copy=False)
    cnn_feat=pca.fit_transform(np_cnn_feat)
    print('PCA方差率：', pca.explained_variance_ratio_)
    # print(cnn_feat.shape)

    feature = list(zip(feature_frame,cnn_feat.tolist()))
    # print(feature)
    fp=open(filename, 'w')
    for var in feature:
        # print(var[0])
        fp.writelines(str(var))
        fp.write('\n')
    fp.close()



# if __name__=='__main__':
#     vgg_test('/home/detection/PyTorch-YOLOv3/data/1114data','/home/detection/PyTorch-YOLOv3/yolov3_feat/frame_data/ROI/ROI_feat.csv')

