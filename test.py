# coding:utf-8
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, *args):
        super(Net, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        print("forward --->>>")


model1 = nn.Sequential()
model2 = nn.Sequential()
net = Net(model1, model2)
print(net)

