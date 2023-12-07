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
        self.feat = nn.Sequential(nn.Conv2d(in_channel, \
            out_channel, kernel_size=kernel_size, stride=1, \
                padding=padding, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True))
    
    def forward(self, x):
        out = self.feat(x)
        return out

class NetworkInNetwork(nn.Module):
    def __init__(self, num_classes=4, num_stages=3, Nbits=None, symmetric=False, lin_eval_flag=False):
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
        
    
    # need to figure out the next step which is the different amount of blocks
    # also how to make it go down to just 4 classes