# 生成用于神经网络的数据

import time
import copy
import torch
from torch import optim, nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys
from IPython import display
from tqdm import tqdm  # python进度条模块
import warnings
import image  # 导入图像处理模块
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")  # 忽略警告信息
sys.path.append("..")  # 添加路径到搜索路径


# 1、重新自定义DataSet
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.485, 0.456, 0.406])  # 均值
        self.rgb_std = np.array([0.229, 0.224, 0.225])  # 标准差
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        # 定义图像transforms变换
        self.crop_size = crop_size  # （H，W)
        features, labels = image.read_voc_images(root=voc_dir, is_train=is_train,  max_num=max_num)
        # 数据集中可能有图像的尺寸小于随机裁剪指定的输出尺寸
        # 这些样本需要通过自定义的filter函数所移除
        self.features = self.filter(features)  # 滤除大小不符合的样本
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label  # 存储RGB与label映射关系
        print('read ' + str(len(self.features)) + "  ||  " + f'train : {is_train}, val : {not is_train}')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])]
    # PIL图像的size是（W，H）。所以是img.size[1] >= self.crop_size[0]
    # filter函数，只留下高宽都大于输出大小的图像

    def __getitem__(self, idx):
        """重写__getitem__方法"""
        feature, label = image.voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # 虽然PIL图像size是（W,H)  但是get_params方法内部调用了另外方法，自动调整为（H,W)
        return (self.tsf(feature), image.voc_label_indices(label, self.colormap2label))
    # 生成的数据集会包括两个属性，一个是tensor格式的原图数据。
    # 另一个是原图的标签图像经过匹配之后与原图形状相同。但是只有1个通道的每个像素对应类型索引数据

    def __len__(self):
        # 重写__len__方法
        return len(self.features)


# 实例化数据集
batch_size = 32  # 每次读取的数据量
crop_size = (320, 480)  # 指定随机裁剪的输出图像的形状为(320,480)(H, W)
max_num = 20000
# 最多从本地读多少张图片

# 实例化数据集
voc_dir = "./data/VOCdevkit/VOC2012"  # 数据集存储路径
colormap2label = image.colormap2label
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)
x = voc_test[0]


# 设批量大小为32，DataLoader训练集和测试集
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = DataLoader(voc_train, batch_size, shuffle=True,
                        drop_last=True, num_workers=num_workers)
test_iter = DataLoader(voc_test, batch_size, drop_last=True,
                       num_workers=num_workers)
# 测试集没有每经过epoch随机打乱数据
dataloaders = {'train': train_iter, 'val': test_iter}
dataset_sizes = {'train': len(voc_train), 'val': len(voc_test)}

