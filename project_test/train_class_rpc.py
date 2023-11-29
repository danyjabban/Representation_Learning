import time
import os
import threading

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
from torch.distributed import rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

from lars.lars import LARS

from FP_layers import *

from resnet import *



class ResNetCIFAR_shard1(nn.Module):
    def __init__(self, device, num_layers=50, Nbits=None, symmetric=False):
        super(ResNetCIFAR_shard1, self).__init__()
        self._lock = threading.Lock()
        self.device = device
        # assert 0 == 1, "don't use this yet"
        self.num_layers = num_layers
        self.head_conv = nn.Sequential(
            FP_Conv(in_channels=3, out_channels=16,
                      stride=1, padding=1, kernel_size=3, bias=False, Nbits=None, symmetric=symmetric),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = 16
        # Stage 1
        for j in range(num_layers_per_stage):
            strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 16, strides, Nbits=Nbits, symmetric=symmetric))
            num_inputs = 16
        # Stage 2
        for j in range(num_layers_per_stage):
            if j == 0:
                strides = 2
            else:
                strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 32, strides, Nbits=Nbits, symmetric=symmetric))
            num_inputs = 32
            
        self.body_op = nn.Sequential(*self.body_op).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.body_op(x)
        return out.cpu()
    
    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]



class ResNetCIFAR_shard2(nn.Module):
    def __init__(self, device, head_g, num_layers=50, Nbits=None, symmetric=False):
        # assert 0 == 1, "don't use this yet"
        super(ResNetCIFAR_shard2, self).__init__()
        self._lock = threading.Lock()
        self.device = device
        self.num_layers = num_layers

        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = 32
        # Stage 3
        for j in range(num_layers_per_stage):
            if j == 0:
                strides = 2
            else:
                strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 64, strides, Nbits=Nbits, symmetric=symmetric))
            num_inputs = 64

        self.body_op.append(nn.AdaptiveAvgPool2d(1))
        self.body_op = nn.Sequential(*self.body_op).to(self.device)
        self.head_g = head_g.to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            self.features = self.body_op(out)
            self.feat_1d = self.features.mean(3).mean(2)  # this is h=f(*)
            if self.head_g is not None:  # for finetuning and training, 
            # where head isn't separate from the rest of the ResNet. See algorithm 1 in SimCLR paper. 
                self.g_out = self.head_g(self.feat_1d)  # this is z=g(h)=g(f(*))
                # note: g(*) does not reduce # of coordinates to 10, i.e., no logits
            else:  # this is for linear evaluation only, where the head is separate
                self.g_out = self.feat_1d
        return self.g_out.cpu()
    
    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]



class ResNetCIFAR_Dist(nn.Module):
    def __init__(self, num_split, workers, head_g, num_layers=50, Nbits=None, symmetric=False):
        super(ResNetCIFAR_Dist, self).__init__()
        self.num_split = num_split
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetCIFAR_shard1,
            args = ("cuda:0", num_layers, Nbits, symmetric)
        )
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetCIFAR_shard2,
            args=("cuda:1", head_g, num_layers, Nbits, symmetric)
        )
        return

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            x_rref = rpc.RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        return torch.cat(torch.futures.wait_all(out_futures))
    
    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


def train_rpc(num_split, max_epochs, batch_size, lr, reg, head, save_base_path, log_every_n=50):
    def aug_dat(batch):
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
    
    def cifar_dataloader_wo_ddp(bs: int):
        transform_gen = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_gen)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False,
                                                  num_workers=16, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_gen)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
        return trainloader, testloader
    
    def _run_epoch(epoch):

        if epoch >= warmup_iters:
            scheduler = scheduler_after
        else:
            scheduler = scheduler_warmup
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            augment_inputs1 = Trainer_wo_DDP.aug_dat(inputs).to(device)
            augment_inputs2 = Trainer_wo_DDP.aug_dat(inputs).to(device)
            del inputs
            self.optimizer.zero_grad()
            with dist_autograd.context() as context_id:
                outputs1 = model(augment_inputs1)
                outputs2 = model(augment_inputs2)
                del augment_inputs1
                del augment_inputs2
                loss = criterion(outputs1, outputs2)
                del outputs1, outputs2
                dist_autograd.backward(context_id, [loss])
            # self.optimizer.step()
            optimizer.step(context_id)

            train_loss += loss.item()
            total += targets.size(0)
            global_steps += 1

            if global_steps % log_every_n == 0:
                print("[Step=%d]\tLoss=%.4f" % (global_steps, train_loss / (batch_idx + 1)))

            scheduler.step()

        """
        Start the testing code.
        """
        model.eval()
        test_loss = 0
        # correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                augment_inputs1, augment_inputs2 = Trainer_wo_DDP.aug_dat(inputs).to(device), Trainer_wo_DDP.aug_dat(inputs).to(device)
                outputs1, outputs2 = model(augment_inputs1), model(augment_inputs2)
                loss = criterion(outputs1, outputs2)

                test_loss += loss.item()
                total += targets.size(0)
        num_val_steps = len(testloader)
        val_loss = test_loss / num_val_steps
        print("Test Loss=%.4f" % val_loss)
        return

    def _save_checkpoint(epoch: int, save_base_path: str):
        print("Saving...")
        torch.save(self.model.state_dict(), "%s/epoch_%d_bs_%d_lr_%g_reg_%g.pt" 
                                        % (save_base_path, int(epoch), int(batch_size), self.lr, self.reg))
        return
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    model = ResNetCIFAR_Dist(num_split, ["master", "worker1"], head_g=head)
    criterion = ContrastiveLoss(batch_size, temperature=0.5)
    optimizer = DistributedOptimizer(
                        LARS, model.parameter_rrefs(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False
                        )
    
    def create_lr_schheduler(opt_rref):
        return optim.lr_scheduler.LinearLR(opt_rref, start_factor=0.01, end_factor=1.0, 
                                                        total_iters=warmup_iters, verbose=False)

    def lrs_step(lrs_rref):
        lrs_rref.local_value().step()

    lrs_rrefs = []
    for opt_rref in optimizer.remote_optimizers:
        lrs_rrefs = rpc.remote(opt_rref.owner(), create_lr_schheduler, args=(opt_rref,))

    with dist_autograd.context() as context_id:
        # omitting forward-backward-optstep here
        futs = []
        for lrs_rref iin lrs_rrefs:
            futs.append(rpc.rpc_async(lrs_rref.owner(), lrs_step, args=(lrs_rref,)))
        [fut.wait() for fut in futs]
    warmup_iters = 10
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer.optimizer, start_factor=0.01, end_factor=1.0, 
                                                        total_iters=warmup_iters, verbose=False)
    scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, verbose=False)
    trainloader, testloader = Trainer_wo_DDP.cifar_dataloader_wo_ddp(batch_size)
    global_steps = 0

    for epoch in range(max_epochs):
        _run_epoch(epoch)
        # only save once on master gpu
        if torch.distributed.get_rani() == 0 and ((epoch+1) % 100 == 0 or epoch == 0):
            _save_checkpoint(epoch, save_base_path)
    # save last epoch
    _save_checkpoint(max_epochs - 1, save_base_path)
    return


def run_worker(rank, world_size, num_split, max_epochs, batch_size, lr, reg, head, save_base_path, log_every_n=50):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        train_rpc(num_split, max_epochs, batch_size, lr, reg, head, save_base_path, log_every_n=log_every_n)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass
    # block until all rpcs finish
    rpc.shutdown()
    return


if __name__ == "__main__":
    # from lars.lars import LARS
    head = nn.Sequential(FP_Linear(64, 64, Nbits=None), nn.ReLU(True), FP_Linear(64, 64, Nbits=None))
    save_base_path = "./saved_models/"
    os.makedirs(save_base_path, exist_ok=True)
    # model = nn.parallel.DistributedDataParallel(ResNetCIFAR(head_g=head))
    # model = nn.DataParallel(ResNetCIFAR(head_g=head))
    batch_size = int(1024)
    world_size = torch.cuda.device_count()
    num_split = 2
    # num_split, batch_size, lr, reg, max_epochs, head, save_base_path, log_every_n
    # tik = time.time() world_size, 1000, batch_size, 0.3*batch_size/256, 1e-6, head, save_base_path, 50,
    mp.spawn(run_worker, args=(world_size, num_split, 1000, batch_size, 0.3*batch_size/256, 1e-6, head, save_base_path, 50,), 
                               nprocs=world_size, join=True)
    # tok = time.time()
    # print(f"number of splits = {num_split}, execution time = {tok - tik}")