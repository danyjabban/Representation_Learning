import time
import os
import argparse
import random

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

from FP_layers import *

from resnet import LinearEvaluation
from train_classes import Trainer_LinEval

torch.manual_seed(0)  # for reproducibility
random.seed(0)  # just in case
np.random.seed(0)  # just in case


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, required=True, 
                        help='which cuda device. On Lab server (tingjun chen), should be in [0,1]')
    parser.add_argument('-e', '--epoch_resnet', type=int, required=True)
    parser.add_argument('-b', '--batch_size_resnet', type=int, required=True)
    # parser.add_argument('-b', '--batchsize', type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:%d' % int(args.device) if torch.cuda.is_available() else 'cpu')

    resnet_model_pth = "./saved_models/epoch_%d_bs_%d_lr_%g_reg_1e-06.pt" % \
                (args.epoch_resnet, args.batch_size_resnet, 0.3*args.batch_size_resnet/256)

    lin_eval_net = LinearEvaluation(method='lin', which_device=device, resnet_model_pth=resnet_model_pth, Nbits=None, symmetric=False).to(device)
    save_base_path = 'saved_models/lin_eval_models'
    resnet_params = {'epoch': args.epoch_resnet, 'bs': args.batch_size_resnet}
    os.makedirs(save_base_path, exist_ok=True)
    # batch_size = int(args.batchsize)
    batch_size = int(512)
    lin_eval_trainer = Trainer_LinEval(model=lin_eval_net, batch_size=batch_size, which_device=device, resnet_params=resnet_params,
                                       lr=0.05, reg=0, log_every_n=50*256/batch_size)
    lin_eval_trainer.train(100, save_base_path)
