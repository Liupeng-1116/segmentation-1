# 定义模型，进行训练

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
import dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")  # 忽略警告信息
sys.path.append("..")  # 添加路径到搜索路径

# 选择设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 21   # 21分类，20种物体加一个背景，共21种像素类别
model_ft = resnet18(pretrained=True)
# 使用ResNet18作为网络主框架。设置True，使用预训练参数

# 特征提取器
for param in model_ft.parameters():
    param.requires_grad = False
    # 首先冻结主框架的参数梯度属性，暂时不学习（更新）它们

model_ft = nn.Sequential(
    *list(model_ft.children())[:-2],
    nn.Conv2d(512, num_classes, kernel_size=1),
    nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32)).to(device)
"""扔掉最后的两层之后，构造一个（32，3， 320，480）的随机数据送入网络得到输出是（32，512，10，15）
也就是经过这些卷积，feature的高宽缩小到原本的1/32
所以要进行上采样恢复原本大小数据
首先是要将通道调整到类别数（21类），通过1*1卷积
再通过转置卷积上采样调整大小。
转置卷积参数是可学习的，默认就是开启权重和偏置参数。
转置卷积的输出大小计算公式见官方文档
https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html?highlight=convtranspose#torch.nn.ConvTranspose2d
CSDN一篇文章中：
对于转置卷积层，如果步幅为y​、填充为y/2（假设为整数）、卷积核的⾼和宽为2y，
其他参数按照转置卷积默认（主要是dilation=1,output-padd=0,默认）
转置卷积核将输⼊的⾼和宽都将放大y倍

这里设置合适的参数，将高宽放大到原来（放大32倍）。但是元素就不一定是原本的（不是真正的逆卷积还原原本数据）
"""


# 利用双线性插值法，对转置卷积层进行初始化
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor-1
    else:
        center = factor-0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1-abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    weight = torch.Tensor(weight)
    weight.requires_grad = True
    return weight
# 输出weight 的形状是（21， 21， 64， 64）


nn.init.xavier_normal_(model_ft[-2].weight.data, gain=1)  # 1*1 卷积层（倒数第二层）使用Xavier随机初始化
model_ft[-1].weight.data = bilinear_kernel(num_classes, num_classes, 64).to(device)
# 设置插值出的数据为转置卷积核权重初始化参数

dataloaders = dataset.dataloaders
dataset_sizes = dataset.dataset_sizes


def train_model(model: nn.Module, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    val_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    # 每个epoch都有一个训练和验证阶段
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            # train和val的损失以及准确度应该分开放，所以这两个是在第二个for循环内
            # 迭代一个epoch
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):  # 设置只在phase='train'是可用梯度
                    logits = model(inputs)   # [5, 21, 320, 480]
                    loss = criteon(logits, labels.long())
                    # 在https://blog.csdn.net/Fcc_bd_stars/article/details/105158215?spm=1001.2014.3001.5506
                    # 这篇CSDN博客评论区中，提到损失函数计算时标签值转为long型(即torch.int64)，否则报错
                    # 经过测试，确实会报错，必须是int64(long)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((torch.argmax(logits.data, 1)) == labels.data) / (480*320)
                # 一张图像一共有480*320个像素点。所以最终识别正确的概率就是总数除总像素点个数

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()  # 将优化器的状态作为一个字典返回
                }
                torch.save(state, "checkpoint.pth")  # 保存信息

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_losses.append(epoch_loss)
                scheduler.step()

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, val_losses, train_losses, LRs


epochs = 10   # 训练5个epoch
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
# 每3个epochs衰减LR通过设置gamma=0.1
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# 每经过3个epoch，optimizer的学习率降低为原来的十分之一
# 开始训练
model_ft, val_acc_history, train_acc_history, val_losses, train_losses, LRs = train_model(model_ft, criteon, optimizer, exp_lr_scheduler, num_epochs=epochs)

writer = SummaryWriter(log_dir="./train_logs")
model_list = [val_acc_history, train_acc_history, val_losses, train_losses, LRs]
for idx_1, val_1 in enumerate(model_list):
    if idx_1 == 0:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("val_acc_history", val_2, global_step=idx_2)
    if idx_1 == 1:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("train_acc_history", val_2, global_step=idx_2)
    if idx_1 == 2:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("val_losses", val_2, global_step=idx_2)
    if idx_1 == 3:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("train_losses", val_2, global_step=idx_2)
    if idx_1 == 4:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("LRs", val_2, global_step=idx_2)

writer.close()

# 预测
voc_colormap = image.VOC_COLORMAP


def label2image(pred):
    # pred: [320,480]
    colormap = torch.tensor(voc_colormap, device=device, dtype=int)
    x = pred.long()
    return (colormap[x, :]).data.cpu().numpy()   # tensor先上CPU再进行转换为numpy


mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)
writer_img = SummaryWriter("./forecast_logs")


def visualize_model(model: nn.Module, num_images=4):
    was_training = model.training  # 刚开始时模型并没有开启，所以model.training属性是False
    model.eval()  # 开启评估模式
    images_so_far = 0
    # n = num_images
    # imgs = []
    with torch.no_grad():  # 关闭梯度
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device)  # [batch-size,3,320,480]
            outputs = model(inputs)  # 神经网络输出   [batch-size, 21, 320,480]
            pred = torch.argmax(outputs, dim=1)  # [batch-size,320,480],也就是每个像素预测概率最大的那个
            # 取出每个像素预测输出最大的那个的索引，原本（21，320,480)变成（1，320，480）
            inputs_nd = (inputs*std+mean).permute(0, 2, 3, 1)*255
            # 标准化的逆操作，同时将轴更换为（N,H,W，C)。这样得到原本图像

            for j in range(num_images):
                images_so_far += 1
                # pred是（b,320,480)的图像。pred[j]表示每次取一张图像，他的像素预测输出
                pred1 = label2image(pred[j])
                # 最终返回一个(320, 480, 3)的numpy数组。内部数据表示每个像素点处应该给予的颜色
                # pred[j]现在是一个320*480的矩阵。每个元素都是0-20（共21类）中任意整数
                # 也就是，每一个都表示一个索引。比如"0"表示这里预测是背景，应该给予黑色，就在VOC_COLORMAP中根据索引0找到RGB值（0，0，0）
                # 所以最终又会返回一个（320，480，3）的数组（因为转换为numpy,所以通道在后面）
                # imgs += [inputs_nd[j].data.int().cpu().numpy(), pred1, label2image(labels[j])]
                writer_img.add_image(f"{images_so_far}", inputs_nd[j].data.int().cpu().numpy(),
                                     global_step=1, dataformats="HWC")
                writer_img.add_image(f"{images_so_far}", pred1,
                                     global_step=2, dataformats="HWC")
                writer_img.add_image(f"{images_so_far}", label2image(labels[j]),
                                     global_step=3, dataformats="HWC")
                # 每一次添加的是3张图像，原图、自己分割的图、标签图。
                # 一共添加4次，就总共12张图像
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # 固定设置每次只显示4张图了（其实共12张）
                    # image.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
                    writer_img.close()
                    return model.train(mode=was_training)


# 开始验证
visualize_model(model_ft)








