# 包含对数据的预处理等

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

warnings.filterwarnings("ignore")  # 忽略警告信息
sys.path.append("..")  # 添加路径到搜索路径


# 下载数据集,已提前下载
# VOC数据集中用来分割的部分训练和验证一共有2900多个样本。分别存储图像文件名在train.txt和val.txt
# 同时有一个聚集二者的trainval.txt


# 1、取数据，转换PIL格式
def read_voc_images(root="./data/VOCdevkit/VOC2012", is_train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        # 读取训练集数据名
        images = f.read().split()  # 拆分，组成训练集数据名称list
    # print(len(images)) 1464
    if max_num is not None:
        images = images[:min(max_num, len(images))]
        # 按规定的数值取出一定量的数据名称
    features, labels = [None] * len(images), [None] * len(images)
    # print(features)   输出一个空列表（元素全部为None)。
    # [None]本来就是一个新的列表，乘上一次取出的数据量，就是一个新的全None列表
    for i, fname in tqdm(enumerate(images)):
        # 读入jpg格式数据，转为RGB格式的PILImage格式图像（大小不变）
        # features[i] = Image.open(f'{root}/JPEGImages/{fname}.jpg')
        # print(type(features[i])) <class 'PIL.JpegImagePlugin.JpegImageFile'>
        features[i] = Image.open(f'{root}/JPEGImages/{fname}.jpg').convert("RGB")
        # print(type(features[i]))  <class 'PIL.Image.Image'>
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels  # 生成的空列表存储取出的图像
    # PIL image 像素值都在0-255之间


# 这个函数主要为设置jupyter中图像显示问题,pycharm可以不要


# def set_figsize(figsize=(3.5, 2.5)):
#     # 在jupyter使用svg显示
#     display.set_matplotlib_formats('svg')
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize


# 2、显示原图以及标签图像
def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    fig_size = (num_cols * scale, num_rows * scale)
    # 为什么设置scale=2?因为要进行原图像和分割后图像的对比，每次都是上下分别为原图和标签图两个
    _, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


# train_features, train_labels = read_voc_images(max_num=10)
# 一次取出10张训练集图像和标签
n = 5  # 展示5对（原图和标签共10张）
# imgs = train_features[0:n] + train_labels[0:n]  # PIL image
# show_images(imgs, 2, n)

# 3、设置每个类别对应名称以及颜色
# 标签中每个RGB颜色的值
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
# 也就是，我们要将识别出的某种类型物体标成对应颜色
# 比如识别出某一块区域是背景background，就将这一块（RGB）三个通道数值设置为（000），也就是纯黑色


BASE_VECTOR = np.array([256*256, 256, 1])   # 设置一个基向量
# print(VOC_COLORMAP*BASE_VECTOR)   # 与每种类别物体设置的像素颜色值相乘
# 例如：[128,0,0]是飞机的颜色，与基向量相乘之后再求和得到128.
# 记住此时是i=1的飞机指向了128


def build_colormap2label():
    """通过一个一维的numpy数组存储RGB和labels的index之间的映射关系"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)  # 生成一个够大的存储空间，实际感觉并没有什么用
    for i, colormap in enumerate(VOC_COLORMAP*BASE_VECTOR):
        colormap2label[colormap.sum()] = i
        # 还是之前的例子
        # 此时就是在color2label这个一维向量中的i=128处存储了一个值，即1
        # 这个1就是像素颜色设置的索引以及名称索引。
        # 假设一个标签图像上某个像素点，经过与基向量相乘求和之后得到128，就可以根据128找到1
        # 也就是找到了这个像素物体是什么以及这一点应该设置颜色的值
    return colormap2label


colormap2label = build_colormap2label()


def voc_label_indices(colormap, colormap2label):
    """通过RGB获取labels"""
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    colormap = colormap*BASE_VECTOR
    idx = colormap.sum(axis=2)   # 要按通道方向(此时是numpy数组，格式是HWC，所以axis=2)进行求和，三个通道上对应位置加起来是一个像素点
    return colormap2label[idx]
# 假设标签大小是(3，H,W)（3个通道），那么经过idx = colormap.sum(axis=2)之后是H*W个idx
# 也就是会根据这H*W个索引在color2label中找到H*W个位置，取出其中的值
# 生成一个新的(1，H,W)（只有一个通道了现在），这里面存储的是每一个RGB像素点对应的物体类型名以及颜色的索引
# 我们根据这些值就可以找到对应类别以及索引


# y = voc_label_indices(train_labels[0], build_colormap2label())
# print(y[105:115, 130:140])


def voc_rand_crop(feature, label, height, width):
    """
    随机裁剪原图和标签图，但是二者裁剪出来的区域必须保持一致。
    也就是要用到transforms.RandomCrop的get_params方法，获取到一个随机的裁剪区域坐标参数
    为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
    Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    # 返回裁剪区域的左上右下四个元素坐标
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    # 随机生成的坐标送入crop方法，进行裁剪
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label

# 显示1对原图及标签，随机裁剪5次的图像和标签
# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
    # 将一张图像和标签随机裁剪5次，统一裁剪成200*300
# show_images(imgs[::2] + imgs[1::2], 2, n)
# 因为是将一张图片和标签裁剪5次。所以imgs列表中存储有10张图片
# 其中原图，从0（索引）开始，0、2、4....所以[::2] 只要设置间隔为2就行
# 同理，但是标签是从1开始，1\3\5...所以设置为[1::2]











