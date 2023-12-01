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

from resnet import LinearEvaluation


class Trainer_wo_DDP():
    def __init__(
        self, model,
        batch_size, lr, reg, train_for_finetune: int, log_every_n=50, write=True
    ) -> None:
        super().__init__()
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        self.model = model
        
        self.acc_steps = max(1, int(batch_size / 1024))
        # https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
        # https://www.blog.dailydoseofds.com/p/gradient-accumulation-increase-batch#:~:text=This%20technique%20works%20because%20accumulating,explicitly%20increase%20the%20batch%20size.
        self.batch_size = int(batch_size / self.acc_steps)
        self.train_for_finetune = train_for_finetune  # == 0 means not training for finetune
        assert self.batch_size == np.round(batch_size / self.acc_steps)
        self.lr = lr
        self.reg = reg
        self.log_every_n = log_every_n

        self.criterion = ContrastiveLoss(batch_size, temperature=0.5)
        self.optimizer = LARS(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.reg, nesterov=False)
        self.warmup_iters = 10
        self.scheduler_warmup = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, 
                                                            total_iters=self.warmup_iters, verbose=False)
        self.scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, verbose=False)
        self.trainloader, self.testloader = Trainer_wo_DDP.cifar_dataloader_wo_ddp(self.batch_size, self.train_for_finetune)
        self.global_steps = 0

        self.write = write
        # self.scaler = torch.cuda.amp.GradScaler()
        return

    def _run_epoch(self, epoch):

        if epoch >= self.warmup_iters:
            scheduler = self.scheduler_after
        else:
            scheduler = self.scheduler_warmup
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            augment_inputs1 = Trainer_wo_DDP.aug_dat(inputs).to(device)
            augment_inputs2 = Trainer_wo_DDP.aug_dat(inputs).to(device)
            del inputs
            
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs1 = self.model(augment_inputs1)
            outputs2 = self.model(augment_inputs2)
            del augment_inputs1
            del augment_inputs2
            loss = self.criterion(outputs1, outputs2)
            loss.backward()
            isnan = sum(torch.isnan(torch.tensor(torch.cat((outputs1.clone().detach(), outputs2.clone().detach())).clone().detach())).to('cpu').numpy().astype(int).flatten())
            del outputs1, outputs2
            # self.scaler.scale(loss).backward()
            if isnan != 0: 
                print(isnan)
            # self.optimizer.step()
            if (batch_idx+1) % self.acc_steps == 0 or (batch_idx+1 == len(self.trainloader)):
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()
                # self.scaler.step(self.optimizer)
            # self.scaler.update()

            train_loss += loss.item()
            total += targets.size(0)
            self.global_steps += 1

            if self.global_steps % (self.log_every_n * self.acc_steps) == 0:
                print("[Step=%d]\tLoss=%.4f" 
                        % (self.global_steps, train_loss / (batch_idx + 1)))
        # end training for this epoch

        """
        Start the testing code.
        """
        self.model.eval()
        test_loss = 0
        # correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                augment_inputs1, augment_inputs2 = Trainer_wo_DDP.aug_dat(inputs).to(device), Trainer_wo_DDP.aug_dat(inputs).to(device)
                outputs1, outputs2 = self.model(augment_inputs1), self.model(augment_inputs2)
                loss = self.criterion(outputs1, outputs2)

                test_loss += loss.item()
                total += targets.size(0)
        num_val_steps = len(self.testloader)
        val_loss = test_loss / num_val_steps
        print("Test Loss=%.4f" % val_loss)
        return val_loss

    def _save_checkpoint(self, epoch: int, save_base_path: str):
        print("Saving...")
        torch.save(self.model.state_dict(), "%s/epoch_%d_bs_%d_lr_%g_reg_%g.pt" 
                                        % (save_base_path, int(epoch), int(self.batch_size * self.acc_steps),
                                           self.lr, self.reg))
        return

    def train(self, max_epochs: int, save_base_path: str):
        file_name = "%s/max_epoch_%d_bs_%d_lr_%g_reg_%g.txt" % (save_base_path, int(max_epochs), int(self.batch_size * self.acc_steps), self.lr, self.reg)
        if file_name not in os.listdir(save_base_path) and self.write:
            f_ptr = open(file_name, 'w')
            f_ptr.close()
        self.optimizer.zero_grad()  # just in case, since I moved self.optimizer.zero_grad() to bottom of 1x iteration 
        # in _run_epoch
        for epoch in tqdm(range(max_epochs), desc='ResNet_bs_%d' % (self.batch_size * self.acc_steps)):
            
            val_loss = self._run_epoch(epoch)  # whether optimizer step happens is determined by batch_idx, not epoch
            if self.write:
                f_ptr = open(file_name, 'a')
                f_ptr.write("%d,%.6f\n" % (epoch, val_loss))
                f_ptr.close()
            # only save once on master gpu
            if (epoch+1) % 100 == 0:  # note: since counting from 0 -> when saving add 1.
                self._save_checkpoint(epoch+1, save_base_path)
        # # save last epoch
        # self._save_checkpoint(max_epochs - 1, save_base_path)

    @staticmethod
    def aug_dat(batch):
        """
        Apply transforms once on one batch without reserving space for the others
        """
        augment = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    # transforms.RandomVerticalFlip(p=0.5),  # extra vertical flip?
                                    torchvision.transforms.ColorJitter(0.5)
                                    ])
        # augment(batch) should be size [batch_size*1, 1, 32, 32]
        return augment(batch)
    
    @staticmethod
    def cifar_dataloader_wo_ddp(bs: int, train_for_finetune: int):
        transform_gen = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_gen)
        if train_for_finetune == 1:
            print("training for eventual fine-tuning")
            trainset, _ = torch.utils.data.random_split(trainset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_gen)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
        # print('len(trainloader)', len(trainloader))
        return trainloader, testloader



class Trainer_LinEval(Trainer_wo_DDP):
    def __init__(self, model: LinearEvaluation, which_device, batch_size, resnet_params, lr, reg, log_every_n=50):
        super().__init__(model,  # note: do not use self.model for training
                         batch_size, lr, reg, train_for_finetune=0, log_every_n=log_every_n, write=True)  
        self.lin_eval_model = model  
        self.which_device = which_device

        self.optimizer = optim.SGD(self.lin_eval_model.parameters(), lr=self.lr, 
                                momentum=0.875, weight_decay=self.reg, nesterov=True)
        self.criterion = nn.CrossEntropyLoss()
        self.start = time.time()
        self.resnet_params = resnet_params
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(epochs*0.5), 
        #                                                 int(epochs*0.75)], gamma=0.1)
        
        # currently not sharing _run_epoch and _save_checkpoint and train
    
    def _save_checkpoint(self, save_base_path: str):
        print("Saving...")
        torch.save(self.lin_eval_model.state_dict(), "%s/Resnet(epoch_%d_bs_%d)_LinEval(lr_%g_reg_%g_method_%s).pt" 
                                        % (save_base_path, int(self.resnet_params['epoch']), 
                                           int(self.resnet_params['bs']), self.lr, self.reg, self.lin_eval_model.method))
        return

    def _run_epoch(self, epoch):
        """
        Start the training code.
        """
        self.lin_eval_model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.which_device), targets.to(self.which_device)
            self.optimizer.zero_grad()
            outputs = self.lin_eval_model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.global_steps += 1

            if self.global_steps % self.log_every_n == 0:
                end = time.time()
                num_examples_per_second = self.log_every_n * self.batch_size / (end - self.start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (self.global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                self.start = time.time()

        self.scheduler.step()

        """
        Start the testing code.
        """
        self.lin_eval_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.which_device), targets.to(self.which_device)
                outputs = self.lin_eval_model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(self.testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))
        return test_loss / (num_val_steps), val_acc

    def train(self, max_epochs: int, save_base_path: str):
        best_acc = 0
        self.start = time.time()
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs, verbose=False)
        file_name = "%s/Resnet(epoch_%d_bs_%d)_LinEval(lr_%g_reg_%g_method_%s).txt" \
                    % (save_base_path, int(self.resnet_params['epoch']), 
                        int(self.resnet_params['bs']), self.lr, self.reg, self.lin_eval_model.method)
        if file_name not in os.listdir(save_base_path) and self.write:
            f_ptr = open(file_name, 'w')
            f_ptr.close()
        self.optimizer.zero_grad()  # just in case, since I moved self.optimizer.zero_grad() to bottom of 1x iteration 
        # in _run_epoch
        for epoch in tqdm(range(max_epochs), desc='ResNet_lin_eval_bs_%d' % (self.batch_size * self.acc_steps)):
            val_loss, val_acc = self._run_epoch(epoch)  # whether optimizer step happens is determined by batch_idx, not epoch
            if self.write:
                f_ptr = open(file_name, 'a')
                f_ptr.write("%d,%.6f,%.6f\n" % (epoch, val_loss, val_acc))
                f_ptr.close()
            # only save once on master gpu
            if best_acc < val_acc:  # note: since counting from 0 -> when saving add 1.
                best_acc = val_acc
                self._save_checkpoint(save_base_path)