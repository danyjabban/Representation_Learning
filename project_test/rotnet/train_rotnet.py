import argparse
import random
import os
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FP_layers import *
from rotnet_pytorch import NetworkInNetwork

from train_classes_rotnet import RotNetTrainer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('-b', '--batchsize', type=int, required=True)
    # parser.add()
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_base_path = './save_models/RotNet_logs'
    os.makedirs(save_base_path, exist_ok=True)
    model = NetworkInNetwork(4).to(device)
    trainer = RotNetTrainer(model=model, batch_size=128, 
                            device=device, lr=.1, reg=5e-4,
                            momentum=.9, log_every_n=50, nesterov=True,
                            write=True)
    trainer.train(max_epochs=200, save_base_path=save_base_path)