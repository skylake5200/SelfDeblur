import torch
import torch.nn as nn
from .common import *
# from .non_local_embedded_gaussian import NONLocalBlock2D
# from .non_local_concatenation import NONLocalBlock2D
# from .non_local_gaussian import NONLocalBlock2D
from .non_local_dot_product import NONLocalBlock2D


# 0.020s taken for {skip}
def skip(num_input_channels=2, num_output_channels=3,
         num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
         filter_size_down=3, filter_size_up=3, filter_skip_size=1,
         need_sigmoid=True, need_bias=True,
         pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
    """
    利用跳跃连接来构建编解码网络
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    # Python assert用于判断一个表达式，在表达式条件为 false 的时候触发异常
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    # num_channels_down [16, 32, 64, 128, 128]
    n_scales = len(num_channels_down)

    # 这里是啥意思
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        # upsample_mode = ['nearest', 'nearest','nearest','nearest','nearest']
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        # downsample_mode = ['stride', 'stride', 'stride', 'stride', 'stride']
        downsample_mode = [downsample_mode] * n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        # filter_size_down = [3, 3, 3, 3, 3]
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        # filter_size_up = [3, 3, 3, 3, 3]
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1  # last_scale: 4

    cur_depth = None

    '''
    - nn.Sequential()
    - torch的核心是Module类
    - Sequential继承自Module，通常可以看作为一种容器，Sequential类不同的实现(三种实现)，参见：
    - https://blog.csdn.net/qq_27825451/article/details/90551513
    '''
    model = nn.Sequential()
    model_tmp = model
    sign = True
    input_depth = num_input_channels  # input_depth： 8

    # i: 0, 1, 2, 3, 4
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()
        # num_channels_skip=[4, 4, 4, 4, 4]
        if num_channels_skip[i] != 0:
            '''
            Sequential(
                (1): Concat(
                    (0): Sequential()
                    (1): Sequential()
                )
            )
            '''
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        '''
        Sequential(
            (1): Concat(
                (0): Sequential()
                (1): Sequential()
            )
            (2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        '''
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        '''
        Sequential(
            (1): Concat(
                (0): Sequential(
                    (1): Sequential(
                        (0): ReflectionPad2d((0, 0, 0, 0))
                        (1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
                    )
                    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (3): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (1): Sequential()
            )
            (2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        '''
        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        '''
        Sequential(
            (1): Concat(
                (0): Sequential(
                    (1): Sequential(
                        (0): ReflectionPad2d((0, 0, 0, 0))
                        (1): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
                    )
                    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (3): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (1): Sequential(
                    (1): Sequential(
                        (0): ReflectionPad2d((1, 1, 1, 1))
                        (1): Conv2d(8, 128, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (3): LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
            (2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        '''
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        '''
        Sequential(
            (1): Concat(
                (0): Sequential(
                    (1): Sequential(
                        (0): ReflectionPad2d((0, 0, 0, 0))
                        (1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
                    )
                    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (3): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (1): Sequential(
                    (1): Sequential(
                        (0): ReflectionPad2d((1, 1, 1, 1))
                        (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (3): LeakyReLU(negative_slope=0.2, inplace=True)
                    (4): NONLocalBlock2D(
                        (g): Sequential(
                            (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
                            (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
                        )
                        (W): Sequential(
                            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
                            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        )
                        (theta): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
                        (phi): Sequential(
                            (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
                            (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
                        )
                    )
                )  
            )
            (2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        '''
        if i > 1:
            deeper.add(NONLocalBlock2D(in_channels=num_channels_down[i]))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        # if sign:
        #     print("model_temp is ", model_tmp)
        #     sign = False
        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    # for end
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
