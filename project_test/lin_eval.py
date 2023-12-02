import time
import os
import argparse
import random

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression

from FP_layers import *

from resnet import *
from train_classes import *

torch.manual_seed(0)  # for reproducibility
random.seed(0)  # just in case
np.random.seed(0)  # just in case


def collate_data(model):
    # trainloader, _ = Trainer_wo_DDP.cifar_dataloader_wo_ddp(bs=512, train_for_finetune=0, use_default=0)
    # # use_default=0 -> trainloader uses all augmentations
    # _, testloader = Trainer_wo_DDP.cifar_dataloader_wo_ddp(bs=512, train_for_finetune=0, use_default=1)
    # # use_default=1 -> testloader does not use augmentations

    trainloader, testloader = Trainer_wo_DDP.cifar_dataloader_wo_ddp(bs=512, train_for_finetune=0, use_default=1)
    traindata_lst, trainlbl_lst = [], []
    # for _, (batch, _, labels) in enumerate(trainloader):
    for _, (batch, labels) in enumerate(trainloader):
        traindata_lst.extend(model(batch.to(device)).clone().detach().cpu().numpy())
        trainlbl_lst.extend(labels.cpu().numpy())
    testdata_lst, testlbl_lst = [], []
    for _, (batch, labels) in enumerate(testloader):
        testdata_lst.extend(model(batch.to(device)).clone().detach().cpu().numpy())
        testlbl_lst.extend(labels.cpu().numpy())
    return np.array(traindata_lst), np.array(trainlbl_lst), np.array(testdata_lst), np.array(testlbl_lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, required=True, 
                        help='which cuda device. On Lab server (tingjun chen), should be in [0,1]')
    parser.add_argument('-e', '--epoch_resnet', type=int, required=True)
    parser.add_argument('-b', '--batch_size_resnet', type=int, required=True)
    parser.add_argument('-l', '--lr_resnet', type=float, required=True)
    parser.add_argument('-m', '--embed_dim', type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:%d' % int(args.device) if torch.cuda.is_available() else 'cpu')

    # base_path = "./saved_models/w_data_normalise_nesterovTrue/"
    base_path = "./saved_models/wo_data_normalise/"
    resnet_model_pth = base_path + "epoch_%d_bs_%d_lr_%g_reg_1e-06_embedDim_%d.pt" % \
                (args.epoch_resnet, args.batch_size_resnet, float(args.lr_resnet), args.embed_dim)
    model = ResNetCIFAR(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(resnet_model_pth))
    X_train, y_train, X_test, y_test = collate_data(model)
    logistic = LogisticRegression(n_jobs=-1, max_iter=250).fit(X_train, y_train)
    print(logistic.score(X_test, y_test))

    # lin_eval_net = LinearEvaluation(embed_dim=args.embed_dim, method='lin', which_device=device, resnet_model_pth=resnet_model_pth, 
    #                                 Nbits=None, symmetric=False).to(device)
    # lin_eval_save_base_path = base_path + 'lin_eval_models/'
    # resnet_params = {'epoch': args.epoch_resnet, 'bs': args.batch_size_resnet, 'lr': args.lr_resnet, 
    #                  'embed_dim': args.embed_dim}
    # os.makedirs(lin_eval_save_base_path, exist_ok=True)
    # batch_size = int(512)
    # lin_eval_trainer = Trainer_LinEval(model=lin_eval_net, batch_size=batch_size, which_device=device, resnet_params=resnet_params,
    #                                    lr=0.1*batch_size/256, reg=0, log_every_n=50*256/batch_size)
    # lin_eval_trainer.train(100, lin_eval_save_base_path)

    
