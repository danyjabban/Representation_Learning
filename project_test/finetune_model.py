from resnet_pytorch import ResNet_PyTorch_wrapper
import torch
import torch.nn as nn
import torch.functional as F

class FinetuneResNetPyTorch(ResNet_PyTorch_wrapper):
    def __init__(self, load_path, embed_dim=128, Nbits=None, symmetric=False):
        super(FinetuneResNetPyTorch, self).__init__(embed_dim=embed_dim, Nbits=Nbits, symmetric=symmetric)
        self.num_classes = 10
        self.fc_to_10 = nn.Linear(self.embed_dim, self.num_classes, bias=True)
        self.load_state_dict(torch.load(load_path), strict=False)
        

    def forward(self, x):
        x = self.f(x)
        self.features = torch.flatten(x, start_dim=1)
        self.g_out = F.normalize(self.g(self.features), dim=-1)
        return self.fc_to_10(self.g_out)
