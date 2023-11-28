import time
import os

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# from torch.trainers import Trainer

from lars.lars import LARS

from FP_layers import *

from resnet import ResNetCIFAR, ContrastiveLoss


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12325"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return


class TrainerDDP():
    def __init__(
        self, gpu_id, model,
        batch_size, lr, reg, log_every_n=50
    ) -> None:
        # super().__init__(gpu_id, model, trainloader, testloader)
        super().__init__()
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        # self.model = DDP(ResNetCIFAR(head_g=head, num_layers=50), device_ids=[gpu_id])
        self.gpu_id = gpu_id
        self.model = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(), device_ids=[self.gpu_id], output_device=self.gpu_id)
        self.batch_size = batch_size
        self.lr = lr
        self.l2_reg = reg
        self.log_every_n = log_every_n

        self.criterion = ContrastiveLoss(batch_size, temperature=0.5)
        self.optimizer = LARS(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.l2_reg, nesterov=False)
        self.warmup_iters = 10
        self.scheduler_warmup = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, 
                                                            total_iters=self.warmup_iters, verbose=True)
        self.scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, verbose=True)
        cifar_dataloader_ddp(self.batch_size)
        return
    
    def cifar_dataloader_ddp(self, bs: int):
        self.sampler_train = DistributedSampler(trainset)
        # transform_train = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),  # transforms happen for each epoch
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        transform_gen = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_gen)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False, sampler=sampler_train, 
                                                num_workers=16)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_gen)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, 
                                                sampler=DistributedSampler(testset, shuffle=False), num_workers=8)
        return trainloader, testloader, sampler_train

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        model_path = self.const["trained_models"] / f"CIFAR10_ddp_epoch{epoch}.pt"
        torch.save(ckp, model_path)
        return

    def _run_epoch(epoch):
        self.trainloader.sampler.set_epoch(epoch)
        # T_0 = epochs -> no restart? Paper uses lin warmup (first 10 epochs) then cosine decay schedule w/o restarts

        global_steps = 0
        start = time.time()

        if epoch >= warmup_iters:
            scheduler = scheduler_after
        else:
            scheduler = scheduler_warmup
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            augment_inputs1 = TrainerDDP.augment_data_separate(inputs).to(device)
            augment_inputs2 = TrainerDDP.augment_data_separate(inputs).to(device)
            del inputs
            self.optimizer.zero_grad()
            outputs1 = self.model(augment_inputs1)
            outputs2 = self.model(augment_inputs2)
            del augment_inputs1
            del augment_inputs2

            isnan = sum(torch.isnan(torch.tensor(torch.cat((outputs1.clone().detach(), outputs2.clone().detach())).clone().detach())).to('cpu').numpy().astype(int).flatten())
            if isnan != 0: 
                print(isnan)
            loss = self.criterion(outputs1, outputs2)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += targets.size(0)
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\t%.1f examples/second" 
                        % (global_steps, train_loss / (batch_idx + 1), num_examples_per_second))
                start = time.time()

            scheduler.step()

        """
        Start the testing code.
        """
        self.model.eval()
        test_loss = 0
        # correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                augment_inputs1, augment_inputs2 = TrainerDDP.augment_data_separate(inputs).to(device), TrainerDDP.augment_data_separate(inputs).to(device)
                outputs1, outputs2 = self.model(augment_inputs1), self.model(augment_inputs2)
                loss = self.criterion(outputs1, outputs2)

                test_loss += loss.item()
                total += targets.size(0)
        num_val_steps = len(testloader)
        val_loss = test_loss / num_val_steps
        print("Test Loss=%.4f" % val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
        return

    def _save_checkpoint(epoch):
        print("Saving...")
        torch.save(self.model.state_dict(), "test_model_epoch_%d_bs_%d_lr_%g_reg_%g.pt" 
                                        % int(epoch, self.batch_size, self.lr, self.l2_reg))
        return

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            self.sampler_train.set_epoch(epoch)
            self._run_epoch(epoch)
            # only save once on master gpu
            if self.gpu_id == 0 and (epoch+1) % 100 == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(max_epochs - 1)


    # def test(self, final_model_path: str):
    #     self.model.load_state_dict(
    #         torch.load(final_model_path, map_location="cpu")
    #     )
    #     self.model.eval()

    #     with torch.no_grad():
    #         for src, tgt in self.testloader:
    #             src = src.to(self.gpu_id)
    #             tgt = tgt.to(self.gpu_id)
    #             out = self.model(src)
    #             self.valid_acc.update(out, tgt)
    #     print(
    #         f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}%"
    #     )
    #     return
    
    @staticmethod
    def augment_data_separate(batch):
        """
        Apply transforms once on one batch without reserving space for the others
        """
        augment = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),  # extra vertical flip?
                                    torchvision.transforms.ColorJitter(0.5)
                                    ])
        # augment(batch) should be size [batch_size*1, 1, 32, 32]
        return augment(batch)

def run_main(rank, world_size, total_epochs, batch_size, lr, reg, head, log_every_n):
    ddp_setup(rank, world_size)
    model = ResNetCIFAR(head_g=head)
    trainer = TrainerDDP(rank, model, batch_size, lr, reg, log_every_n=log_every_n)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    # from lars.lars import LARS
    head = nn.Sequential(FP_Linear(64, 64, Nbits=None), nn.ReLU(True), FP_Linear(64, 64, Nbits=None))
    # model = nn.parallel.DistributedDataParallel(ResNetCIFAR(head_g=head))
    # model = nn.DataParallel(ResNetCIFAR(head_g=head))
    batch_size = int(1024)
    world_size = torch.cuda.device_count()
    mp.spawn(run_main, args=(world_size, 1000, batch_size, 0.3*batch_size/256, 1e-6, head, 50,), nprocs=world_size)