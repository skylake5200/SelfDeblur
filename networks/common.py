import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    '''
        nn.Sequential().add_module(name, module)
        代码参见：
        https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model/50345328#50345328
        def add_module(self, module):
            self.add_module(str(len(self) + 1), module)
        torch.nn.Module.add = add_module
    '''


torch.nn.Module.add = add_module


# 连接网络的容器
class Concat(nn.Module):
    def __init__(self, dim, *args):
        # Concat(1, skip, deeper)
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    '''
    python __call__() 函数
    只要实现__call__函数，可以把该类型的对象当作函数来调用，可以认为__call__函数重载了()运算符
    for example: net = Net(1, 10, 1) net(x), 这里net(x)相当于直接调用了forward函数
    '''
    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3: diff3 + target_shape3])

        # torch.cat(seq, dim, out=None)
        # 其中seq表示要连接的两个序列，以元组的形式给出，例如：seq = (a, b)
        # dim=1纵向连接, dim=0横向连接
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    # 下采样
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)

    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    # filter(函数，序列)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])

    return nn.Sequential(*layers)

