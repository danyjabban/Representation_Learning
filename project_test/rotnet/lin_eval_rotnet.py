import torch.nn as nn
import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FP_layers import *
from data_utils import *
from rotnet_pytorch import NetworkInNetwork
from train_classes_rotnet import RotNetLinEvalTrainer
from rotnet_nonlinclass import NonLinearClassifier

torch.manual_seed(0)  # for reproducibility
random.seed(0)  # just in case
np.random.seed(0)  # just in case

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--fname', type=str, required=True)
    parser.add_argument('-k', '--k_img_per_cat', type=int, required=True)
    parser.add_argument('-f', '--freeze', type=int, required=True)
    parser.add_argument('-e', '--epoch_rotnet', type=int, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-l', '--lin_eval', type=str, required=True)
    parser.add_argument('-n', '--base_path', type=str, required=True)
    parser.add_argument('-t', '--type_non_linear', type=str, required=True)
    args = parser.parse_args()
    lr = .1
    if args.epoch_rotnet > 60:
        lr /= 5
    if args.epoch_rotnet > 120:
        lr /= 5
    if args.epoch_rotnet > 160:
        lr /= 5
    print(f'lr: {lr}')
    rotnet_model_pth = f'{args.base_path}/epoch_{args.epoch_rotnet}_bs_{args.batch_size}_lr_{lr}_reg_0.0005.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_base_path = './save_models/RotNet_LinEval_logs'
    os.makedirs(save_base_path, exist_ok=True)
    lin_eval_flag = args.lin_eval
    k_img_per_cat = args.k_img_per_cat
    # breakpoint()
    rotnet_model = NetworkInNetwork(4,lin_eval_flag=lin_eval_flag).to(device)
    rotnet_model.load_state_dict(torch.load(rotnet_model_pth))
    if args.freeze == 0:
        freeze = False
    else:
        freeze = True
    rotnet_params = {'epoch': args.epoch_rotnet, 'bs': args.batch_size, 'lr': .1}
    if args.type_non_linear == 'fc':
        type_class = 'mult_fc'
        if lin_eval_flag == '1':
            nChannels = 96*16*16
        if lin_eval_flag == '2' or lin_eval_flag == '3' or lin_eval_flag == '4':
            nChannels = 192*8*8
    elif args.type_non_linear == 'conv':
        if lin_eval_flag == '1':
            nChannels = 96
        if lin_eval_flag == '2' or lin_eval_flag == '3' or lin_eval_flag == '4':
            nChannels = 192
        type_class = 'NIN_conv'
    
    classifier_model = NonLinearClassifier(type_class=type_class, num_classes=10,
                                           nChannels=nChannels).to(device)
    trainer = RotNetLinEvalTrainer(rotnet_model=rotnet_model, classifier_model=classifier_model,
                                   rotnet_params=rotnet_params, batch_size=128,
                                   device=device, lin_eval_type=lin_eval_flag,
                                   nonlin=args.type_non_linear, k_img_per_cat=k_img_per_cat,
                                   lr=.1, reg=5e-4, momentum=.9, log_every_n=50, 
                                   nesterov=True, freeze_rotnet=freeze)
    
    trainer.train(max_epochs=100, save_base_path=save_base_path)