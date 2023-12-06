from resnet_pytorch import ResNet_PyTorch_wrapper
from train_classes import Trainer_FineTune

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os

class FinetuneResNetPyTorch(ResNet_PyTorch_wrapper):
    def __init__(self, load_path, embed_dim=128, Nbits=None, symmetric=False):
        super(FinetuneResNetPyTorch, self).__init__(embed_dim=embed_dim, Nbits=Nbits, symmetric=symmetric)
        self.num_classes = 10
        self.fc_to_10 = nn.Linear(2048, self.num_classes, bias=True)
        self.load_state_dict(torch.load(load_path), strict=False)
        

    def forward(self, x):
        x = self.f(x)
        self.features = torch.flatten(x, start_dim=1)
        return self.fc_to_10(self.features)


if __name__ == '__main__':
    f_ptr = open("gpu_ids.txt")  # this reads the GPU on which each model was trained 
    lines = f_ptr.readlines()
    f_ptr.close()
    gpu_dict = {}
    for line in lines:
       (key, val) = line.split(',')
       gpu_dict[int(key)] = int(val)

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_resnet', type=int, required=True)
    parser.add_argument('-b', '--batch_size_resnet', type=int, required=True)
    parser.add_argument('-l', '--lr_resnet', type=float, required=True)
    parser.add_argument('-m', '--embed_dim', type=int, required=True) # resnet parameters

    parser.add_argument('-s', '--batch_size_finetune', type=int, required=True)  # batch size for fine tune == 4096
    parser.add_argument('-p', '--labelled_perc', type=float, required=True)  # percentage of train data with whith to finetune
    parser.add_argument('-f', '--finetune_ep', type=int, required=True)  # number of epochs to fine tune
    args = parser.parse_args()

    device = torch.device('cuda:%d' % int(gpu_dict[args.batch_size_resnet]) if torch.cuda.is_available() else 'cpu')    
    resnet_base_path = "./saved_models/PyTorchResNet_woDatNormalise/"
    finetune_base_path = resnet_base_path + "finetune/"  # where the model/txt files are saved

    os.makedirs(finetune_base_path, exist_ok=True)
    resnet_model_pth = resnet_base_path + "epoch_%d_bs_%d_lr_%g_reg_1e-06_embedDim_%d.pt" % \
                                (args.epoch_resnet, args.batch_size_resnet, float(args.lr_resnet), args.embed_dim)
    model_finetune = FinetuneResNetPyTorch(resnet_model_pth, embed_dim=args.embed_dim, Nbits=None, symmetric=False).to(device)
    # don't need to load since FinetuneResNetPyTorch loads it for you
    batch_size = int(args.batch_size_finetune)
    resnet_params = {'epoch': args.epoch_resnet, 'bs': args.batch_size_resnet, 'lr': args.lr_resnet, 
                     'embed_dim': args.embed_dim}
    finetuner = Trainer_FineTune(model=model_finetune, which_device=device, batch_size=batch_size, lr=0.05*batch_size/256, 
                                 reg=0, resnet_params=resnet_params, device=device,
                                 labelled_perc=args.labelled_perc, log_every_n=max(1, int(256/batch_size * 50)), write=True, 
                                 save_base_path=finetune_base_path)
    finetuner.train(max_epochs=args.finetune_ep)
