import torch
import torch.nn as nn
import torch.nn.functional as F

from FP_layers import *

class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides, Nbits=None, symmetric=False):
        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Sequential(
            FP_Conv(in_channels=in_chs, out_channels=out_chs,
                      stride=strides, padding=1, kernel_size=3, bias=False, Nbits=Nbits, symmetric=symmetric),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            FP_Conv(in_channels=out_chs, out_channels=out_chs,
                      stride=1, padding=1, kernel_size=3, bias=False, Nbits=Nbits, symmetric=symmetric),
            nn.BatchNorm2d(out_chs))

        if in_chs != out_chs:
            self.id_mapping = nn.Sequential(
                FP_Conv(in_channels=in_chs, out_channels=out_chs,
                          stride=strides, padding=0, kernel_size=1, bias=False, Nbits=Nbits, symmetric=symmetric),
                nn.BatchNorm2d(out_chs))
        else:
            self.id_mapping = None
        self.final_activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None:
            x_ = self.id_mapping(x)
        else:
            x_ = x
        return self.final_activation(x_ + out)

class ResNetCIFAR(nn.Module):
    def __init__(self, head_g, num_layers=50, Nbits=None, symmetric=False):
        super(ResNetCIFAR, self).__init__()
        self.num_layers = num_layers
        self.head_conv = nn.Sequential(
            FP_Conv(in_channels=3, out_channels=16,
                      stride=1, padding=1, kernel_size=3, bias=False, Nbits=None, symmetric=symmetric),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = 16
        # Stage 1
        for j in range(num_layers_per_stage):
            strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 16, strides, Nbits=Nbits, symmetric=symmetric))
            num_inputs = 16
        # Stage 2
        for j in range(num_layers_per_stage):
            if j == 0:
                strides = 2
            else:
                strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 32, strides, Nbits=Nbits, symmetric=symmetric))
            num_inputs = 32
        # Stage 2
        for j in range(num_layers_per_stage):
            if j == 0:
                strides = 2
            else:
                strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 64, strides, Nbits=Nbits, symmetric=symmetric))
            num_inputs = 64
            
        self.body_op = nn.Sequential(*self.body_op)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.head_g = FP_Linear(64, 10, Nbits=None)
        self.head_g = head_g

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        self.features = self.avg_pool(out)
        self.feat_1d = self.features.mean(3).mean(2)  # this is h=f(*)
        if self.head_g is not None:  # for finetuning only, where head isn't separate
            self.head_g = self.head_g(self.feat_1d)  # this is z=g(h)=g(f(*))
            return self.head_g  # note: g(*) does not reduce # of coordinates to 10, i.e., no logits
        else:  # this is for linear evaluation only, where the head is separate
            self.head_g = self.feat_1d
            return self.head_g


class LinearEvaluation(nn.Module):
    def __init__(self, method, resnet_model_pth, Nbits=None, symmetric=False):
        # three heads: identity mapping; (2) linear projection; and (3) nonlinear projection
        # nonlinear: W2 * ReLu(W1 * h), where h=ResNet(*)=f(*)
        self.method = method
        self.valid_method = {'identity': None, 
                            'lin': FP_Linear(64, 64, Nbits=None), 
                            'nonlin': nn.Sequential(FP_Linear(64, 64, Nbits=None), 
                                                    nn.ReLU(True),  # in-place ReLU
                                                    FP_Linear(64, 64, Nbits=None))
                            }
        self.to_logits = FP_Linear(64, 10, Nbits=None)  # linear eval needs logistic regression
        # output of self.head_g will always be 

        self.head_g = self.valid_method[self.method]
        assert self.method in self.valid_methods.keys()
        self.model = ResNetCIFAR(head_g=None, Nbits=Nbits, symmetric=symmetric)
        # head_g = None -> model should return embeddings h=f(*) right after avg pool
        self.model.load_state_dict(torch.load(resnet_model_pth))
        self.embedding = None
        self.logits = None
    
    def forward(self, x):
        with torch.no_grad():
            self.embedding = self.model(x).clone().detatch().requires_grad_(True)
        if self.head_g is not None: # linear or nonlinear projections
            self.logits = self.to_logits(self.head_g(embedding))
        else:  # identity mapping
            self.logits = self.to_logits(embedding)
        return self.logits  # note: use cross entropy loss to do logistic regression
