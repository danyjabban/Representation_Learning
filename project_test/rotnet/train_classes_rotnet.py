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
        # self.trainloader, self.testloader = 
    
    def _run_epoch(self, epoch):
        print(f'\nEpoch: {epoch}')
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (rot_ims, targets) in enumerate(self.trainloader):
            output = self.model(rot_ims.to(self.device))
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
                output = self.model(rot_ims.to(self.device))
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