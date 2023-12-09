import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from FP_layers import *

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(BasicBlock, self).__init__()
        padding = (kernel_size-1)/2
        # make sure to make this work with FP layers later!
        self.feat = nn.Sequential(nn.Conv2d(in_channel, 
                                            out_channel, 
                                            kernel_size=kernel_size, 
                                            stride=1,
                                            padding=padding,
                                            bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True))
    
    def forward(self, x):
        out = self.feat(x)
        return out
    
class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class NetworkInNetwork(nn.Module):
    def __init__(self, num_classes=4, Nbits=None, symmetric=False, lin_eval_flag=False):
        super(NetworkInNetwork, self).__init__()
        nChannels = 192
        nChannels2 = 160
        nChannels3 = 96
        
        self.block1 = nn.Sequential(BasicBlock(3, nChannels, 5),
                                    BasicBlock(nChannels, nChannels2, 1),
                                    BasicBlock(nChannels2, nChannels3, 1),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.block2 = nn.Sequential(BasicBlock(nChannels3, nChannels, 5),
                                    BasicBlock(nChannels,  nChannels, 1),
                                    BasicBlock(nChannels,  nChannels, 1),
                                    nn.AvgPool2d(kernel_size=3,stride=2,padding=1))
        
        self.block3 = nn.Sequential(BasicBlock(nChannels, nChannels, 3),
                                    BasicBlock(nChannels, nChannels, 1),
                                    BasicBlock(nChannels, nChannels, 1))
        
        self.block4 = nn.Sequential(BasicBlock(nChannels, nChannels, 3),
                                    BasicBlock(nChannels, nChannels, 1),
                                    BasicBlock(nChannels, nChannels, 1))
        
        self.globpool = GlobalAveragePooling()
        
        self.linear = nn.Linear(nChannels, num_classes)
        
        self.init_weights()
    
    def forward(self, x):
        x_block1 = self.block1(x)
        x_block2 = self.block2(x_block1)
        x_block3 = self.block3(x_block2)
        x_block4 = self.block4(x_block3)
        x_globpool = self.globpool(x_block4)
        x_linear = self.linear(x_globpool)
        return x_linear
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()