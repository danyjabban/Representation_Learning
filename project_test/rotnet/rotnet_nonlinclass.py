import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class NonLinearClassifier(nn.Module):
    def __init__(self, type_class, num_classes, nChannels, Nbits=None, symmetric=False):
        super(NonLinearClassifier, self).__init__()
        if type_class == 'mult_fc':
            nFeatures = min(num_classes*20, 2048)
            self.classifier = nn.Sequential(Flatten(),
                                            nn.Linear(nChannels, nFeatures, bias=False),
                                            nn.BatchNorm2d(nFeatures),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(nFeatures, nFeatures, bias=False),
                                            nn.BatchNorm2d(nFeatures),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(nFeatures, num_classes))
        elif type_class == 'NIN_conv':
            self.classifier = nn.Sequential(BasicBlock(nChannels, 192, 3),
                                            BasicBlock(192, 192, 1),
                                            BasicBlock(192, 192, 1),
                                            GlobalAveragePooling(),
                                            nn.Linear(192, num_classes))
        
        self.init_weights()
    
    def forward(self, feat):
        return self.classifier(feat)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)