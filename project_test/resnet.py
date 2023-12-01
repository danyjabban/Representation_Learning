import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, num_layers=50, Nbits=None, symmetric=False, lin_eval_key=0):
        lin_eval_key_dict = {0: 'output is returned after passing through self.head_g (e.g., training/fine-tuning)', 
                             1: 'nonlinear head for linear evaluation -> use same head as for training', 
                             2: "don't use default head (linear evaluation with identity or linear mapping)"}
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
        self.head_g = nn.Sequential(FP_Linear(64, 64, Nbits=None), 
                                    nn.ReLU(True), 
                                    FP_Linear(64, 64, Nbits=None))
        self.lin_eval_key = lin_eval_key

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        self.features = self.avg_pool(out)
        self.feat_1d = self.features.mean(3).mean(2)  # this is h=f(*)

        # self.head_g will always be the same nonlinear stuff. 
        # For finetuning and training, use different members
        # instead of changing what the members are.
        self.g_out = self.head_g(self.feat_1d)  # this is z=g(h)=g(f(*))
        return self.g_out  # note: g(*) does not reduce # of coordinates to 10, i.e., no logits
        # if self.lin_eval_key == 0 or self.lin_eval_key == 1:  # lin_eval_key 0 -> not doing lin_eval_key
        #     # for finetuning and training and linear evaluation with nonlinear head
        #     # See algorithm 1 in SimCLR paper. 
        #     self.g_out = self.head_g(self.feat_1d)  # this is z=g(h)=g(f(*))
        #     return self.g_out  # note: g(*) does not reduce # of coordinates to 10, i.e., no logits
        # if self.lin_eval_key == 2:  # this is for linear evaluation with linear/identity head
        #     self.g_out = self.feat_1d
        #     return self.g_out


class LinearEvaluation(nn.Module):
    def __init__(self, method, resnet_model_pth, which_device, Nbits=None, symmetric=False):
        super(LinearEvaluation, self).__init__()
        # three heads: identity mapping; linear projection; and nonlinear projection
        # nonlinear: W2 * ReLu(W1 * h), where h=ResNet(*)=f(*)
        # assert 0 == 1, "don't use this module yet"
        self.method = method
        self.valid_methods = {'identity': 2, # does not pass through head in ResNetCIFAR
                              'lin': 2}  # does not pass through head in ResNetCIFAR
                              #'nonlin': 1}  # does pass through head in ResNetCIFAR
        self.to_logits = nn.Sequential(FP_Linear(64, 10, Nbits=None))  # linear eval needs logistic regression

        self.head_g = nn.Sequential(FP_Linear(64, 64, Nbits=None))
        assert self.method in self.valid_methods.keys()

        self.resnet = ResNetCIFAR(Nbits=Nbits, symmetric=symmetric, lin_eval_key=self.valid_methods[self.method]).to(which_device)
        # head_g = None -> model should return embeddings h=f(*) right after avg pool
        self.resnet.load_state_dict(torch.load(resnet_model_pth))
        self.embedding = None
        self.logits = None
    
    def forward(self, x):
        with torch.no_grad():
            # _ = self.resnet(x).clone().detach().requires_grad_(True)
            # self.embedding = self.resnet.feat_1d.clone().detach()

            self.embedding = self.resnet(x).clone().detach()
            # print(self.embedding.shape)
        if self.method == 'linear': # linear projections
            self.logits = self.to_logits(self.head_g(self.embedding))
        else:  # identity mapping or nonlinear projection. This should've already been specified in ResNetCIFAR
            self.logits = self.to_logits(self.embedding)
        return self.logits  # note: use cross entropy loss to do logistic regression


# https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
# vectorised version
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, emb_i.size(0))
        sim_ji = torch.diag(similarity_matrix, -emb_i.size(0))
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        if emb_i.size(0) != self.batch_size:
            nominator = torch.exp(positives / self.temperature)
            denominator = (~torch.eye(emb_i.size(0)*2, emb_i.size(0)*2, dtype=bool)).float().to(emb_i.device) * torch.exp(similarity_matrix / self.temperature)
        else: 
            nominator = torch.exp(positives / self.temperature)
            denominator = self.negatives_mask.to(emb_i.device) * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1) + 1e-8)
        loss = torch.sum(loss_partial) / (2 * emb_i.size(0))
        # print(torch.norm(emb_i.clone().detach(), p=2), torch.norm(emb_j.clone().detach(), p=2))
        return loss
