import torch
import torch.nn as nn
import torch.nn.functional as F

from FP_layers import *
from resnet_pytorch import ResNet_PyTorch_wrapper


class LinearEvaluation(nn.Module):
    def __init__(self, model: ResNet_PyTorch_wrapper):
        super(LinearEvaluation, self).__init__()
        self.resnet = model
        if self.resnet.lin_eval_flag:
            self.head_g = FP_Linear(2048, 512, Nbits=None)  # if using ResNet.feat_1d, then use this line
            self.to_logits = FP_Linear(512, 10, Nbits=None)  # linear eval needs logistic regression
        else:
            self.head_g = FP_Linear(self.resnet.embed_dim, 512, Nbits=None)  # if using ResNet.feat_1d, then use this line
            self.to_logits = FP_Linear(512, 10, Nbits=None)  # linear eval needs logistic regression
        
    def forward(self, x):
        # with torch.no_grad():
            # _ = self.resnet(x)
            # self.embedding = self.resnet.features.clone().detach()
            # print(self.embedding.shape)
        self.embedding = self.resnet(x).clone().detach()
        # print(sum(np.isnan(self.embedding.clone().cpu().numpy()).flatten()))
        self.logits = self.to_logits(self.head_g(self.embedding))
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
