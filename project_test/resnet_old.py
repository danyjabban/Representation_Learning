import torch.nn as nn

from FP_layers import *

class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides, Nbits=None, symmetric=False):
        # assert 0 == 1, "don't use this yet"
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
    def __init__(self, embed_dim, num_layers=50, Nbits=None, symmetric=False, lin_eval_flag: bool = False):

        # assert 0 == 1, "don't use this yet"
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
        self.embed_dim = embed_dim
        self.lin_eval_flag = lin_eval_flag
        self.head_g = nn.Sequential(FP_Linear(64, 2048, Nbits=None), 
                                    nn.ReLU(True), 
                                    FP_Linear(2048, self.embed_dim, Nbits=None))

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        self.features = self.avg_pool(out)
        self.feat_1d = self.features.mean(3).mean(2)  # this is h=f(*)
        if self.lin_eval_flag:
            return self.feat_1d
        # for lin_eval/finetuning, use self.lin_eval_flag == True
        self.g_out = self.head_g(self.feat_1d)  # this is z=g(h)=g(f(*))
        return self.g_out  # note: g(*) does not reduce # of coordinates to 10, i.e., no logits