from train_classes import Trainer_Vanilla
from resnet_pytorch import ResNet_PyTorch_vanilla_wrapper

import torch

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, required=True)
    args = parser.parse_args()
    device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
    save_base_path = "./saved_models/vanilla_resnet/"
    os.makedirs(save_base_path, exist_ok=True)
    
    train_bs = 128
    val_bs = 100
    lr = 0.1
    data_path = 'data'

    model = ResNet_PyTorch_vanilla_wrapper().to(device) 
    trainer = Trainer_Vanilla(model=model, device=device, 
                              train_bs=train_bs, val_bs=val_bs, 
                              lr=lr, reg=1e-4, save_base_path=save_base_path, 
                              data_path=data_path)
    trainer.train(max_epochs=200)