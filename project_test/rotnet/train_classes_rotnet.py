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
                 lr=.1, reg=5e-4, momentum=.9, log_every_n=50, 
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
        if epoch == 60 or epoch == 120 or epoch == 160:
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


class RotNetLinEvalTrainer():
    def __init__(self, rotnet_model: NetworkInNetwork, 
                 classifier_model: NonLinearClassifier,
                 rotnet_params, batch_size, device,
                 lin_eval_type, nonlin, k_img_per_cat=5000,
                 lr=.1, reg=5e-4, momentum=.9, log_every_n=50,
                 nesterov=True, freeze_rotnet=True):
        super().__init__()
        self.rotnet_model = rotnet_model
        self.classifier_model = classifier_model
        self.rotnet_params = rotnet_params
        self.batch_size = batch_size
        self.device = device
        self.freeze_rotnet = freeze_rotnet
        self.lr = lr
        self.lin_eval_type = lin_eval_type
        self.nonlin = nonlin
        self.k_img_per_cat = k_img_per_cat
        self.reg = reg
        self.momentum = momentum
        self.log_every_n = log_every_n
        self.nesterov = nesterov
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.freeze_rotnet:
            self.optimizer = optim.SGD(
                params=self.classifier_model.parameters(), lr=self.lr, 
                momentum=self.momentum, weight_decay=self.reg, 
                nesterov=self.nesterov)
        else:
            self.optimizer = optim.SGD(
                params=list(self.classifier_model.parameters())+
                list(self.rotnet_model.parameters()),
                lr=self.lr, momentum=self.momentum,
                weight_decay=self.reg, nesterov=self.nesterov)
        self.global_steps = 0
        
        self.trainloader, self.testloader = self.cifar_dataloader_lineval_rotnet()
    
    def _save_checkpoint(self, save_base_path: str):
        print('Saving...')
        location = f"{save_base_path}/Rotnet(epoch_{int(self.rotnet_params['epoch'])}_bs_{int(self.rotnet_params['bs'])}_lr_{float(self.rotnet_params['lr'])})_LinEval(nonlin_{self.nonlin}_type_{self.lin_eval_type}_lr_{self.lr}_reg_{self.reg}).pt"
        torch.save(self.classifier_model.state_dict(), location)
    
    def _run_epoch(self, epoch):
        # breakpoint()
        if self.nonlin == 'fc':
            if epoch == 20 or epoch == 40 or epoch == 45:
                self.lr = self.lr / 5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f'Decreasing learning rate by a factor of 5: {self.lr}')
        else:
            if epoch == 35 or epoch == 70 or epoch == 85:
                self.lr = self.lr / 5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f'Decreasing learning rate by a factor of 5: {self.lr}')

        
        print(f'\nEpoch: {epoch}')
        if self.freeze_rotnet:
            self.rotnet_model.eval()
        else:
            self.rotnet_model.train()
        self.classifier_model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.freeze_rotnet:
                with torch.no_grad():
                    rotnet_features = self.rotnet_model(inputs)
                outputs = self.classifier_model(rotnet_features.clone().detach())
            else:
                rotnet_features = self.rotnet_model(inputs)
                outputs = self.classifier_model(rotnet_features)
            # breakpoint()
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.global_steps += 1
            
            if self.global_steps % self.log_every_n == 0:
                print(f"[Step={self.global_steps}]\tLoss={(train_loss/(batch_idx+1)):.4f}\tAcc={(correct/total):.4f}")
        
        if not self.freeze_rotnet:
            self.classifier_model.eval()
            
        self.rotnet_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                rotnet_features = self.rotnet_model(inputs)
                outputs = self.classifier_model(rotnet_features)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(self.testloader)
        val_acc = correct / total
        print(f'Test Loss = {(test_loss/num_val_steps):.4f}, Test Acc = {val_acc:.4f}')
        return (test_loss / (num_val_steps)), val_acc        
        
    
    def train(self, max_epochs: int, save_base_path: str):
        best_acc = 0
        self.optimizer.zero_grad()
        for epoch in tqdm(range(max_epochs), desc='RotNet_lin_eval'):
            val_loss, val_acc = self._run_epoch(epoch)
            if best_acc < val_acc:
                best_acc = val_acc
                self._save_checkpoint(save_base_path)
        print(f'Best Test Accuracy: {best_acc}')
            
    def cifar_dataloader_lineval_rotnet(self):
        # TODO: fine tuning subsample thing!
        #
        trainset = CIFAR10_train_rotnet_lin_eval(root='./data', train=True)
        testset = CIFAR10_train_rotnet_lin_eval(root='./data', train=False)
        
        if self.k_img_per_cat != 5000:
            label_idx_map = {}
            labels = trainset.targets
            for idx, label in enumerate(labels):
                if label not in label_idx_map:
                    label_idx_map[label] = []
                label_idx_map[label].append(idx)
            all_idx = []
            for cat in label_idx_map.keys():
                label_idx_map[cat] = label_idx_map[cat][:self.k_img_per_cat]
                all_idx += label_idx_map[cat]
            all_idx = sorted(all_idx)
            # can also use this
            # x = Subset(trainset, all_idx)
            trainset.data = trainset.data[all_idx]
            new_labels = []
            for idx in all_idx:
                new_labels.append(labels[idx])
            trainset.targets = new_labels
            
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=16, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=8, pin_memory=True)
        return trainloader, testloader
    
    