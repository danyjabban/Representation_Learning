import time
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from FP_layers import * 

from data_utils import *

from rotnet_pytorch import NetworkInNetwork

from rotnet_nonlinclass import NonLinearClassifier

class RotNetTrainer():
    def __init__(self, model: NetworkInNetwork, batch_size, device,
                 lr=.1, reg=5e-4, momentum=.9, log_every_n=5, 
                 nesterov=True, write=True) -> None:
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.reg = reg
        self.log_every_n = log_every_n
        self.write = write
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=lr, 
                                   momentum=momentum, weight_decay=reg, 
                                   nesterov=nesterov)
        self.global_steps = 0
        # do train loader test loader here
        self.trainloader, self.testloader = self.cifar_dataloader_rotnet()
    
    def _run_epoch(self, epoch):
        print(f'\nEpoch: {epoch}')
        if epoch == 30 or epoch == 60 or epoch == 80:
            self.lr = self.lr / 5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            print(f'Decreasing learning rate by a factor of 5: {self.lr}')
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (rot_ims, targets) in enumerate(self.trainloader):
            # breakpoint()
            rot_ims, targets = rot_ims.to(device), targets.to(device)
            bs, rots, chan, h, w = rot_ims.shape
            rot_ims = rot_ims.view([bs*rots, chan, h, w])
            targets = targets.view([bs*rots])
            output = self.model(rot_ims)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            total += targets.size(0)
            self.global_steps += 1
            
            if self.global_steps % self.log_every_n == 0:
                print(f'[Step={self.global_steps}\tLoss={(train_loss/(batch_idx+1)):.4f}]')
        
        self.model.eval()
        test_loss = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (rot_ims, targets) in enumerate(self.testloader):
                rot_ims, targets = rot_ims.to(device), targets.to(device)
                bs, rots, chan, h, w = rot_ims.shape
                rot_ims = rot_ims.view([bs*rots, chan, h, w])
                targets = targets.view([bs*rots])
                output = self.model(rot_ims)
                loss = self.criterion(output, targets)
                test_loss += loss.item()
                total += targets.size(0)
        num_val_steps = len(self.testloader)
        val_loss = test_loss / num_val_steps
        print(f'Test Loss={val_loss:.4f}')
        return val_loss
    
    def _save_checkpoint(self, epoch: int, save_base_path: str):
        print('Saving...')
        location = f'{save_base_path}/epoch_{int(epoch)}_bs_{int(self.batch_size)}_lr_{self.lr}_reg_{self.reg}.pt'
        torch.save(self.model.state_dict(), 
                   location)
        
    def train(self, max_epochs: int, save_base_path: str):
        file_name = f'{save_base_path}/max_epoch_{int(max_epochs)}_bs_{int(self.batch_size)}_lr_{self.lr}_reg_{self.reg}.txt'
        if file_name.split('/')[-1] not in os.listdir(save_base_path) and self.write:
            f_ptr = open(file_name, 'w')
            f_ptr.close()
        self.optimizer.zero_grad()
        for epoch in tqdm(range(max_epochs), desc=f'RotNet_bs_{self.batch_size}'):
            val_loss = self._run_epoch(epoch)
            if self.write:
                f_ptr = open(file_name, 'a')
                f_ptr.write(f'{epoch},{val_loss:.6f}\n')
                f_ptr.close()
            if (epoch+1) % 10 == 0:
                self._save_checkpoint(epoch+1, save_base_path)
    
    def cifar_dataloader_rotnet(self):
        trainset = CIFAR10_RotNet(root='./data', train=True)
        testset = CIFAR10_RotNet(root='./data', train=False)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=16, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=8, pin_memory=True)
        return trainloader, testloader


class Trainer_LinEval():
    def __init__(self, rotnet_model: NetworkInNetwork, 
                 classifier_model: NonLinearClassifier,
                 )