import time
import os
import argparse
import random

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

from lars.lars import LARS

from FP_layers import *

from resnet import ResNetCIFAR, ContrastiveLoss

from tqdm import tqdm

from train_classes import Trainer_wo_DDP

torch.manual_seed(0)  # for reproducibility
random.seed(0)  # just in case
np.random.seed(0)  # just in case


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, required=True, 
                        help='which cuda device. On Lab server (tingjun chen), should be in [0,1]')
    parser.add_argument('-b', '--batchsize', type=int, required=True)
    parser.add_argument('-f', '--train_for_finetune', type=int, required=True, help="=0 means normal trianing, "
                                        "=1 means train for enventual fine-tuning")
    args = parser.parse_args()

    valid_bs = {256, 512, 1024, 2048, 4096}
    assert args.batchsize in valid_bs

    device = torch.device('cuda:%d' % int(args.device) if torch.cuda.is_available() else 'cpu')
    # from lars.lars import LARS
    # head = nn.Sequential(FP_Linear(64, 64, Nbits=None), nn.ReLU(True), FP_Linear(64, 64, Nbits=None))
    save_base_path = "./saved_models/"
    os.makedirs(save_base_path, exist_ok=True)
    # model = nn.parallel.DistributedDataParallel(ResNetCIFAR(head_g=head))
    # model = nn.DataParallel(ResNetCIFAR(head_g=head))

    batch_size = int(args.batchsize)

    model = ResNetCIFAR().to(device)
    trainer = Trainer_wo_DDP(model, batch_size, lr=0.3*batch_size/256, reg=1e-6, 
                             train_for_finetune=args.train_for_finetune, log_every_n=int(256/batch_size * 50))
    trainer.train(total_epochs=1000, save_base_path=save_base_path)