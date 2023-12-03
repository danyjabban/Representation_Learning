import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit, symmetric=False):
        '''
        symmetric: True for symmetric quantization, False for asymmetric quantization
        '''
        if bit is None:
            wq = w
        elif bit == 0:
            wq = w * 0
        else:
            # Build a mask to record position of zero weights
            np_weights = w.detach().cpu().numpy()
            weight_mask = np.ma.masked_equal(x=np_weights, value=0)  # zero will be marked true
            weight_mask = torch.tensor(1 - np.ma.getmask(weight_mask).astype(int)).to(device)  
            # zero will be marked zero, weight_mask onto GPU
            
            flattened_np_weights = np_weights.flatten()
            
            # Lab3 (a), Your code here:
            if symmetric == False:
                max_, min_ = np.max(flattened_np_weights), np.min(flattened_np_weights)
                # Compute alpha (scale) for dynamic scaling
                alpha = max_ - min_
                # Compute beta (bias) for dynamic scaling
                beta = min_
                # Scale w with alpha and beta so that all elements in ws are between 0 and 1
                ws = (w - beta) / alpha
                
                step = (2 ** (bit)) - 1
                # Quantize ws with a linear quantizer to "bit" bits
                R = (1/step) * torch.round(step * ws)  # r_o = 1/(2^k-1) * round((2^k-1)*r_i), assume r_i=r_in=ws
                # Scale the quantized weight R back with alpha and beta
                wq = alpha * R + beta
            
            # Lab3 (e), Your code here:
            else:  # if symmetric == True
                # In symmetric quantization, the quantization levels are symmetric to zero.
                abs_max_ = np.max(np.abs(flattened_np_weights))
                alpha = abs_max_
                beta = 0  # do not shift
                
                # Scale w with alpha and beta=0 so that all elements in ws are between -0.5 and 0.5
                ws = ((w - beta) / alpha)
                
                step = (2 ** (bit - 1)) - 1
                
                # Quantize ws with a linear quantizer to "bit" bits
                R = (1/step) * torch.round(step * ws)
                # Scale the quantized weight R back with alpha and beta
                wq = alpha * R + beta
                pass

            # Restore zero elements in wq 
            wq = wq * weight_mask
            
        return wq

    @staticmethod
    def backward(ctx, g):
        return g, None, None  # my addition. Originally return g, None

class FP_Linear(nn.Module):
    def __init__(self, in_features, out_features, Nbits=None, symmetric=False):
        super(FP_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.Nbits = Nbits
        self.symmetric = symmetric
        
        # Initailization
        m = self.in_features
        n = self.out_features
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        return F.linear(x, STE.apply(self.linear.weight, self.Nbits, self.symmetric), self.linear.bias)

    

class FP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1,
                 stride=1, padding=0, bias=False, Nbits=None, symmetric=False):
        super(FP_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, 
                              bias=bias, dilation=dilation)
        self.Nbits = Nbits
        self.symmetric = symmetric

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        return F.conv2d(x, STE.apply(self.conv.weight, self.Nbits, self.symmetric), self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

    



