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

from resnet_pytorch import *
from train_classes import *

torch.manual_seed(0)  # for reproducibility
random.seed(0)  # just in case
np.random.seed(0)  # just in case


def collate_data(model, device, verbose=False):
    # trainloader, _ = Trainer_wo_DDP.cifar_dataloader_wo_ddp(bs=512, train_for_finetune=0, use_default=0)
    # # use_default=0 -> trainloader uses all augmentations
    # _, testloader = Trainer_wo_DDP.cifar_dataloader_wo_ddp(bs=512, train_for_finetune=0, use_default=1)
    # # use_default=1 -> testloader does not use augmentations

    trainloader, testloader = Trainer_wo_DDP.cifar_dataloader_wo_ddp(bs=512, train_for_finetune=0, use_default=1)
    traindata_lst, trainlbl_lst = [], []
    for idx, (batch, labels) in enumerate(trainloader):
        with torch.no_grad():
            feat = model(batch.to(device))
        traindata_lst.extend(feat.clone().detach().cpu().numpy())
        trainlbl_lst.extend(labels.cpu().numpy())
        if verbose:
            print("train idx %d", idx)
    testdata_lst, testlbl_lst = [], []
    for idx, (batch, labels) in enumerate(testloader):
        with torch.no_grad():
            feat = model(batch.to(device))
        testdata_lst.extend(feat.clone().detach().cpu().numpy())
        testlbl_lst.extend(labels.cpu().numpy())
        if verbose:
            print("test idx %d" % idx)
    return np.array(traindata_lst), np.array(trainlbl_lst), np.array(testdata_lst), np.array(testlbl_lst)


if __name__ == "__main__":
    f_ptr = open("gpu_ids.txt")  # this reads the GPU on which each model was trained 
    lines = f_ptr.readlines()
    f_ptr.close()
    gpu_dict = {}
    for line in lines:
       (key, val) = line.split(',')
       gpu_dict[int(key)] = int(val)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fname', type=str, required=True)
    parser.add_argument('-e', '--epoch_resnet', type=int, required=True)
    parser.add_argument('-b', '--batch_size_resnet', type=int, required=True)
    parser.add_argument('-l', '--lr_resnet', type=float, required=True)
    parser.add_argument('-m', '--embed_dim', type=int, required=True)
    args = parser.parse_args()

    # device = torch.device('cuda:%d' % int(args.device) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:%d' % int(gpu_dict[args.batch_size_resnet]) if torch.cuda.is_available() else 'cpu')
    base_path = "./saved_models/PyTorchResNet_woDatNormalise/"  # where to find the resnet models
    resnet_model_pth = base_path + "epoch_%d_bs_%d_lr_%g_reg_1e-06_embedDim_%d.pt" % \
                (args.epoch_resnet, args.batch_size_resnet, float(args.lr_resnet), args.embed_dim)
    # which resnet model to use
    
    model = ResNet_PyTorch_wrapper(embed_dim=args.embed_dim, lin_eval_flag=True).to(device)
    model.load_state_dict(torch.load(resnet_model_pth))
    print('successfully loaded')

    X_train, y_train, X_test, y_test = collate_data(model, device)
    logistic = LogisticRegression(n_jobs=-1, max_iter=250).fit(X_train, y_train)
    fptr = open(args.fname, 'a')
    fptr.write('%d,%d,%g,%d,%.8f\n' % (args.batch_size_resnet, args.epoch_resnet, args.lr_resnet, args.embed_dim, logistic.score(X_test, y_test)))
    fptr.close()
    # lin_eval_net = LinearEvaluation(model=model).to(device)
    # lin_eval_save_base_path = base_path + 'lin_eval_models/'
    # resnet_params = {'epoch': args.epoch_resnet, 'bs': args.batch_size_resnet, 'lr': args.lr_resnet, 
    #                  'embed_dim': args.embed_dim}
    # os.makedirs(lin_eval_save_base_path, exist_ok=True)
    # batch_size = int(512)
    # lin_eval_trainer = Trainer_LinEval(model=lin_eval_net, batch_size=batch_size, which_device=device, resnet_params=resnet_params,
    #                                    lr=0.1*batch_size/256, reg=0, log_every_n=50*256/batch_size)
    # lin_eval_trainer.train(100, lin_eval_save_base_path)
