# 常用工具
import torch
import torchvision
import cv2
import time
'''
    # Python Imaging Library，已经是Python平台实际上的图像处理标准库了
    # 原始PIL只支持到python 2.7，开源社区在PIL的基础上创建了兼容的版本，名字叫Pillow
    # 支持python 3.x，我们可以直接安装Pillow
    # Image模块是PIL中最重要的模块，它有一个类叫做image，与模块名称相同
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


def crop_image(img, d=32):
    '''
        Make dimensions divisible by d
        :param img:
        :param d:
        :return:
    '''

    imgsize = img.shape
    new_size = (imgsize[0] - imgsize[0] % d,
                imgsize[1] - imgsize[1] % d)
    bbox = [
            int((imgsize[0] - new_size[0])/2),
            int((imgsize[1] - new_size[1])/2),
            int((imgsize[0] + new_size[0])/2),
            int((imgsize[1] + new_size[1])/2),
    ]

    img_cropped = img[0:new_size[0], 0:new_size[1], :]
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters()]
        elif  opt=='down':
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """
    Load PIL image
    对于彩色图像，不管其图像格式是PNG，还是BMP，或者JPG，在PIL中，使用Image模块的open()函数打开后，返回的图像对象的模式都是"RGB"
    而对于灰度图像，不管其图像格式是PNG，还是BMP，或者JPG，打开后，其模式为"L"
    PIL中有九种不同模式，分别为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F
    具体请参见：https://blog.csdn.net/icamera0/article/details/50843172
    """
    img = Image.open(path)
    return img


# 0.006s taken for {get_image}
def get_image(path, imsize=-1):
    """
    Load an image and resize to a specific size.
    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)  # <PIL.PngImagePlugin.PngImageFile image mode=L size=255x255 at 0x1E8D36C94A8>
    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)
            print("img.resize(imsize, Image.ANTIALIAS) --->>> ", img)
    # Converts image in PIL format to np.array
    img_np = pil_to_np(img)

    return img, img_np

def fill_noise(x, noise_type):
    """
    Fills tensor `x` with noise of type `noise_type`
    """
    # 相当于直接在原tensors上改变，无需返回
    if noise_type == 'u':
        x.uniform_()  # 从连续均匀分布中采样的数字，均匀分布
    elif noise_type == 'n':
        x.normal_()  # 正态分布
    else:
        assert False


# 0.000s taken for {get_noise}
def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """
    Returns a pyTorch Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    meshgrid函数就是用两个坐标轴上的点在平面上画网格(传入的参数是两个的时候)
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 均匀分布 'n' for normal 正态分布
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        # 如果spatial_size是一个整数的话(没有给出维度)，就进行转换
        spatial_size = (spatial_size, spatial_size)

    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        # 制作张量
        net_input = torch.zeros(shape)
        # 填充噪声，图像的话
        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1), np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    '''
        Converts image in PIL format to np.array.
        From W x H x C [0...255] to C x W x H [0..1]
        当使用PIL.Image.open()打开图片后，如果要使用img.shape函数，需要先将image形式转换成array数组
        img = numpy.array(im)
    '''
    ar = np.array(img_PIL)  # ar.shape (255, 255)
    # 这一步在做什么
    if len(ar.shape) == 3:
        # transpose在不指定参数是默认是矩阵转置
        '''
        详情请参见：
        https://blog.csdn.net/u012762410/article/details/78912667?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
        '''
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]  # ar.shape (1, 255, 255)

    # PIL图像对象，其type都是np.array，array里的元素类型(dtype)均为np.uint8
    # numpy.astype()方法改变元素类型(可以变成np.float32)
    return ar.astype(np.float32) / 255.


def np_to_pil(img_np): 
    '''
        Converts image in np.array format to PIL image.
        From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=[5000, 10000, 15000], gamma=0.1)  # learning rates
        for j in range(num_iter):
            scheduler.step(j)
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def readimg(path_to_image):
    img = cv2.imread(path_to_image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    return img, y, cb, cr

