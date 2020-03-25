# 在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用
# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，也就是它在该版本中不是语言标准，那么我如
# 果想要使用的话就需要从future模块导入
from __future__ import print_function
# 由于torch是非自动求导的，每一层的梯度的计算必须用net:backward才能计算gradInput和网络中的参数的梯度
import torch
import torch.optim
import torch.nn as nn
import matplotlib.pyplot as plt
# argparse模块可以让人轻松编写用户友好的命令行接口
import argparse
import logging
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import glob
# scikit-image是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，与matlab一样
from skimage.io import imread
from skimage.io import imsave
from skimage import img_as_ubyte
import warnings
# tqdm是一个快速，可扩展的Python进度条，可以在 Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
from tqdm import trange
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
from network_show import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 使用argparse的第一步就是创建ArgumentParser对象
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levin/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='frequency to save results')
opt = parser.parse_args()

# cuDNN使用非确定性算法
# 如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法，False则禁用
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pytorch的数据类型为各式各样的Tensor，Tensor可以理解为高维矩阵，与Numpy中的Array类似
# Pytorch中的tensor又包括CPU上的数据类型和GPU上的数据类型，一般GPU上的Tensor是CPU上的Tensor加cuda()函数得到
# 通过使用Type函数可以查看变量类型，一般系统默认是torch.FloatTensor类型
# 例如data = torch.Tensor(2,3)是一个2*3的张量，类型为FloatTensor; data.cuda()就转换为GPU的张量类型，torch.cuda.FloatTensor类型
'''
    Torch定义了七种CPU tensor类型和八种GPU tensor类型，详情请参见：
    https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/
    ！注意： 会改变tensor本身的函数操作会用一个下划线_后缀来标示
'''
dtype = torch.cuda.FloatTensor
'''
一般情况下，pytorch调用GPU通过.cuda()函数，但是也可以用如下这种方式：
https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/2
'''
method = 'noise'
warnings.filterwarnings("ignore")

# 取数据阶段，通过路径获取图像
# opt.data_path is "datasets/levin/"
files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

train_n_img = 0

# start image
for f in files_source:
    train_n_img += 1
    # padding--->SAME和VALID
    '''
    - 在tensorflow中，tf.pad()的作用是填充
        pad(
            tensor,
            paddings,
            mode='CONSTANT',
            name=None
        )
        其中mode可以取三个值，分别是"CONSTANT" ,"REFLECT","SYMMETRIC"
        - mode="CONSTANT"是常数填充，一般填充0
        - mode="REFLECT"是映射填充，上下填充顺序和paddings是相反的，左右顺序补齐
        - mode="SYMMETRIC"是对称填充，上下填充顺序是和paddings相同的，左右对称补齐
    '''
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    # 噪声标准差
    reg_noise_std = 0.001
    # 单幅图像路径
    path_to_image = f
    '''
        os.path 模块主要用于获取文件的属性，详情请参考
        https://www.runoob.com/python/python-os-path.html
    '''
    imgname = os.path.basename(f)  # return image name

    # 这里的[0]直接取了图像名，裁剪后的格式(image_name, extension)
    imgname = os.path.splitext(imgname)[0]  # 分割路径，返回路径名和文件扩展名的元组

    # 调整kernel size，返回-1表示未找到
    if imgname.find('kernel1') != -1:
        opt.kernel_size = [17, 17]
    if imgname.find('kernel2') != -1:
        opt.kernel_size = [15, 15]
    if imgname.find('kernel3') != -1:
        opt.kernel_size = [13, 13]
    if imgname.find('kernel4') != -1:
        opt.kernel_size = [27, 27]
    if imgname.find('kernel5') != -1:
        opt.kernel_size = [11, 11]
    if imgname.find('kernel6') != -1:
        opt.kernel_size = [19, 19]
    if imgname.find('kernel7') != -1:
        opt.kernel_size = [21, 21]
    if imgname.find('kernel8') != -1:
        opt.kernel_size = [21, 21]

    # load image and convert to np. -1表示没有调整大小
    # 这里做了一个转换，从PIL对象到np array，然后再转换为Tensor
    # imgs是np array类型
    # 这里开始拿到图像数据，而不仅仅是文件名 size = (255, 255)
    _, imgs = get_image(path_to_image, -1)  # imgs.shape (1, 255, 255)
    # numpy.array 转换为 torch.Tensor，这里的y为真实值ground truth，y为tensor类型
    y = np_to_torch(imgs).type(dtype)
    # img_size (1, 255, 255)
    img_size = imgs.shape

    # ######################################################################
    # padding，保证原图卷积和大小不变，padding为(k - 1) / 2
    # 这里为什么不除以二呢？
    padh, padw = opt.kernel_size[0] - 1, opt.kernel_size[1] - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padh, img_size[2] + padw

    '''
    x_net:用于生成图像？y = k ⊙ x + n
    '''
    # 输入深度为什么是8？试出来的？
    input_depth = 8

    # 当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整
    # 或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要
    # 使用detach()函数来切断一些分支的反向传播
    # 将参数从网络中分离出来，不参与梯度更新
    # method指定图像或者模糊核初始化的方式，包括noise或者meshgrid
    # net_input torch.Size([1, 8, 271, 271])
    net_input = get_noise(input_depth, method, (opt.img_size[0], opt.img_size[1])).type(dtype).detach()

    net = skip( input_depth, 1,
                num_channels_down=[128, 128, 128, 128, 128],
                num_channels_up=[128, 128, 128, 128, 128],
                num_channels_skip=[16, 16, 16, 16, 16],
                upsample_mode='bilinear',  # 双线性的
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    # 图像网络Gx
    net = net.type(dtype)

    # 模型可视化(pdf)
    # make_dot(net(net_input)).view()

    '''
    k_net: 用于生成模糊核？
    '''
    # 这个200是一个随机码向量，200维，所以也可以说是输入的channels数
    n_k = 200

    # 去掉net_input_kernel = get_noise(n_k, method, (1, 1)).type(dtype).detach()中的.detach()
    # 这里的size为(1, 1)，用于生成模糊核的随机噪声
    # net_input_kernel为输入全连接网络的随机码向量
    net_input_kernel = get_noise(n_k, method, (1, 1)).type(dtype)  # .detach()  # torch.Size([1, 200, 1, 1])

    # squeeze的用法主要就是对数据的维度进行压缩un-squeeze为解压
    net_input_kernel.squeeze_().detach_()  # torch.Size([200])

    # 建立全连接网络，fcn(输入维度=200，输出维度=1，隐藏层个数=1000)
    # 模糊核网络Gk
    net_kernel = fcn(n_k, opt.kernel_size[0] * opt.kernel_size[1])

    # 对model指定了类型cuda
    net_kernel = net_kernel.type(dtype)

    # Losses，均方误差和结构自相似度
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer，使用adam优化器
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': net_kernel.parameters(), 'lr': 1e-4}], lr=LR)

    '''
    torch.optim.lr_scheduler中提供了基于多种epoch数目调整学习率的方法
    功能：
    按设定的间隔调整学习率。这个方法适合后期调试使用，观察loss曲线，为每个实验定制学习率调整时机
    gamma(float)-学习率调整倍数，默认为0.1，即下降10倍
    milestones(list)- 一个list，每一个元素代表何时调整学习率，list元素必须是递增的
    PyTorch的六个学习率调整方法，详情请参见：
    https://blog.csdn.net/u011995719/article/details/89486359?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    '''
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    '''
    Pytorch Tensor复制
    tensor复制可以使用clone()函数和detach()函数
    clone()函数可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中
    detach()函数可以返回一个完全相同的tensor,新的tensor开辟与旧的tensor共享内存，新的tensor会脱离计算图，不会牵扯梯度计算
    此外，一些原地操作(in-place, such as resize_ / resize_as_ / set_ / transpose_) 在两者任意一个执行都会引发错误
    '''
    # initilization inputs
    # 原：net_input_saved = net_input.detach().clone()
    net_input_saved = net_input.clone().detach()

    '''
    tensor.clone().detach()  New/Shared memory(New)  Still in computation graph(No)
    tensor.clone()  New/Shared memory(New)  Still in computation graph(Yes)
    tensor.detach()  New/Shared memory(Shared)  Still in computation graph(No)
    
    clone提供了非数据共享的梯度追溯功能，而detach又舍弃了梯度功能，因此clone和detach意味着着只做简单的数据复制，
    既不数据共享，也不对梯度共享，从此两个张量无关联
    至于是先clone还是先detach，其返回值一样，一般采用tensor.clone().detach()
    详情参见：https://blog.csdn.net/guofei_fly/article/details/104486708
    '''
    net_input_kernel_saved = net_input_kernel.clone().detach()
    '''
    tqdm主要有两种用法：
    方法一：
        from tqdm import tqdm
        for i in tqdm(range(100)):  # 可以用list
             pass
    方法二：trange(i) 是 tqdm(range(i)) 的简写，效果和效果一一样
        from tqdm import trange
        for i in trange(100):
            pass
    详情请参见：
    https://blog.csdn.net/weixin_44110998/article/details/102696642?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    '''
    # start SelfDeblur
    for step in tqdm(range(num_iter), desc='Training with the %s.png, %d / %d ' % (imgname, train_n_img, len(files_source))):

        # input regularization
        # 输入值不需要梯度？只更新模型中的参数？
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape) \
            .type_as(net_input_saved.data).normal_()

        '''
        torch.type()和torch.type_as()的用法
            - type(new_type=None, async=False) new_type为空，返回本身类型；否则，转换为指定类型
            - type_as(tesnor)，将张量转换为给定类型的张量
        '''
        # net_input_kernel = net_input_kernel_saved + reg_noise_std * torch.zeros(net_input_kernel_saved.shape) \
        # .type_as(net_input_kernel_saved.data).normal_()

        # 动态修改学习率
        scheduler.step(step)
        # optimizer.zero_grad() 原本位置在此，我将其挪动到下面位置

        # get the network output, net_input.grad and net_input_kernel.grad are None.
        # kernel 1, size is (17, 17)
        # out_x shape is torch.Size([1, 1, 271, 271])
        out_x = net(net_input)  # Gx生成图像

        # out_k shape is  torch.Size([289])
        out_k = net_kernel(net_input_kernel)  # Gk生成模糊核

        # view()函数用于改变tensor的形状，view中的-1是自适应的调整，相当于numpy中resize()的功能
        # 这里是将输出的模糊核（一维）转换为矩阵matrix，-1表示自动计算，自适应
        # out_k_m shape is  torch.Size([1, 1, 17, 17])
        out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])

        '''
        torch.nn.functional.conv2d(input, filters, bias, stride, padding, dilation, groups)
            - input，输入图像的大小(mini_batch，in_channels，H，W)，是一个四维tensor，哪怕输入只有一张图像，也需要为此格式
            - filters，卷积核的大小(out_channels，in_channel/groups，H，W)，是一个四维tensor
            - bias，代表每一个channel的bias，是一个维数等于out_channels的tensor
            - stride，是一个数或者一个二元组（SH，SW），代表纵向和横向的步长
            - padding，是一个数或者一个二元组（PH，PW ），代表纵向和横向的填充值
            - dilation，是一个数，代表卷积核内部每个元素之间间隔元素的数目，不常用，默认为0，可以不用管
            - groups，是一个数，代表分组卷积时分的组数，特别的当groups = in_channel时，就是在做逐层卷积(depth-wise conv)
        '''
        # 将两个网络得到的图像与模糊核进行卷积，没加噪声
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        # 迭代步骤小于1000，使用均方误差，大于1000，使用SSIM
        if step < 1000:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1 - ssim(out_y, y)

        # -----------------------------------------
        '''
        pytorch训练固定步骤:
            - optimizer.zero_grad()
            - loss.backward()  计算出每个参数的梯度
            - optimizer.step()  优化器使用学习率来对参数进行优化
        '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # -----------------------------------------

        # 打印训练进度 save_frequency = 100
        if (step + 1) % opt.save_frequency == 0:
            logging.warning("total loss is %f" % total_loss)
            # os.path.join()：将多个路径组合后返回
            save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)

            # out_x_np.shape (1, 271, 271)
            out_x_np = torch_to_np(out_x)
            '''
            - numpy.squeeze()函数
            - 语法：numpy.squeeze(array, axis = None)
                - array表示输入的数组
                - axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错
                - axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目
                - 返回值：数组类型
                - 不会修改原数组
            - 作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
            - 结论：np.squeeze())函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用
            '''
            # out_x_np.shape (271, 271)
            out_x_np = out_x_np.squeeze()

            # 从Python2.2开始，增加操作符 //，result向下取整
            # 做了一个裁剪的操作
            # out_x_np.shape (255, 255)
            out_x_np = out_x_np[padh // 2: padh // 2 + img_size[1], padw // 2: padw // 2 + img_size[2]]

            '''
                原始：imsave(save_path, out_x_np)
                会出现警告：Lossy conversion from float32 to uint8. Range [0, 1]. 
                Convert image to uint8 prior to saving to suppress this warning.
                修改为如下：
                from skimage import img_as_ubyte
                io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_ubyte(img))
            '''
            imsave(save_path, img_as_ubyte(out_x_np))  # 保存图像

            # -------------------------------------------------------------------------------------------
            save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            '''
            - np.max：(a, axis=None, out=None, keepdims=False)
                - 求序列的最值
                - 最少接收一个参数
                - axis：默认为列向(也即 axis=0)，axis = 1 时为行方向的最值
            '''
            out_k_np /= np.max(out_k_np)  # 对模糊核进行一个缩小
            # 保存模糊核
            imsave(save_path, img_as_ubyte(out_k_np))
            '''
            Python在遍历已知的库文件目录过程中，如果见到一个.pth 文件，就会将文件中所记录的路径加入到 sys.path 设置中，于是 .pth 文件说指明的库
            也就可以被 Python 运行环境找到了，python中有一个.pth文件，该文件的用法是：
            首先xxx.pth文件里面会书写一些路径，一行一个
            将xxx.pth文件放在特定位置，则可以让python在加载模块时，读取xxx.pth中指定的路径
            '''
            # 保存模型torch.save
            '''
            保存和加载整个模型:
                torch.save(model_object, 'model.pkl')
                model = torch.load('model.pkl')
            仅保存和加载模型参数(推荐使用)
                torch.save(model_object.state_dict(), 'params.pkl')
                model_object.load_state_dict(torch.load('params.pkl'))
            '''
            torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))

        # Additional Info when using cuda
        # if device.type == 'cuda':
        #     print(torch.cuda.get_device_name(0))
        #     print('Memory Usage:')
        #     print('Allocated: ', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        #     print('Cached: ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

