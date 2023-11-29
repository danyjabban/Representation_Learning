import time
import os

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from lars.lars import LARS

from FP_layers import *

from resnet import ResNetCIFAR, ContrastiveLoss

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return


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
    # return batch  ## debugging only, no augmentation at all


def train_w_DDP(rank, world_size, epochs, batch_size, lr, reg, head, log_every_n=50):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    ddp_setup(rank, world_size)
    net = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(ResNetCIFAR(head_g=head).to(rank)), device_ids=None)
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(rank)
    print('device', rank)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # transforms happen for each epoch
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    best_loss = 1000000  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=False,
                                              sampler=DistributedSampler(trainset),
                                              num_workers=16)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # criterion = nn.CrossEntropyLoss()
    criterion = ContrastiveLoss(batch_size, temperature=0.5)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    optimizer = LARS(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=10, verbose=False)
    scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=False)
    # T_0 = epochs -> no restart? Paper uses lin warmup (first 10 epochs) then cosine decay schedule w/o restarts

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        if epoch >= 10:
            scheduler = scheduler_after
        else:
            scheduler = scheduler_warmup
        trainloader.sampler.set_epoch(epoch)
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        # correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            augment_input1, augment_input2 = augment_data_separate(inputs).to(rank), augment_data_separate(inputs).to(rank)
            del inputs
            optimizer.zero_grad()
            loss = criterion(net(augment_input1), net(augment_input2))
            loss.backward()
            optimizer.step()

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
        net.eval()
        test_loss = 0
        # correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                augment_input1, augment_input2 = augment_data_separate(inputs).to(rank), augment_data_separate(inputs).to(rank)
                # augment_inputs = augment_inputs.to(rank)  # do not targets.to(device)
                loss = criterion(net(augment_inputs1), net(augment_inputs2))
                test_loss += loss.item()
                total += targets.size(0)
        num_val_steps = len(testloader)
        val_loss = test_loss / num_val_steps
        # val_acc = correct / total
        # print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))
        print("Test Loss=%.4f" % val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            if rank == 0:
                print("Saving...")
                torch.save(net.state_dict(), "test_model.pt")
    destroy_process_group()
    return


def train_nt_xet_class(net, epochs, batch_size, lr, reg, log_every_n=50):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    print('device', device)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # transforms happen for each epoch
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    best_loss = 1e5  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # criterion = nn.CrossEntropyLoss()
    # criterion = ContrastiveLossELI5(batch_size, temperature=0.5, verbose=False)
    criterion = ContrastiveLoss(batch_size, temperature=0.5)

    optimizer = LARS(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    warmup_iters = 10
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters, verbose=True)
    scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
    # T_0 = epochs -> no restart? Paper uses lin warmup (first 10 epochs) then cosine decay schedule w/o restarts

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        if epoch >= warmup_iters:
            scheduler = scheduler_after
        else:
            scheduler = scheduler_warmup
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            augment_inputs1 = augment_data_separate(inputs).to(device)
            augment_inputs2 = augment_data_separate(inputs).to(device)
            del inputs
            optimizer.zero_grad()
            outputs1 = net(augment_inputs1)
            outputs2 = net(augment_inputs2)
            del augment_inputs1
            del augment_inputs2

            isnan = sum(torch.isnan(torch.tensor(torch.cat((outputs1.clone().detach(), outputs2.clone().detach())).clone().detach())).to('cpu').numpy().astype(int).flatten())
            if isnan != 0: 
                print(isnan)
            loss = criterion(outputs1, outputs2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\t%.1f examples/second" 
                        % (global_steps, train_loss / (batch_idx + 1), num_examples_per_second))
                # print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                    #   % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        # correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                augment_inputs1, augment_inputs2 = augment_data_separate(inputs).to(device), augment_data_separate(inputs).to(device)
                outputs1, outputs2 = net(augment_inputs1), net(augment_inputs2)
                loss = criterion(outputs1, outputs2)

                test_loss += loss.item()
                total += targets.size(0)
        num_val_steps = len(testloader)
        val_loss = test_loss / num_val_steps
        print("Test Loss=%.4f" % val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            if epoch % 100 == 0 and epoch > 0:
                print("Saving...")
                torch.save(net.state_dict(), "test_model_epoch_%d_bs_%d_lr_%g_reg_%g.pt" % int(epoch, batch_size, lr, reg))
    return


def finetune(net, epochs, batch_size, lr, reg, log_every_n=50):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)

    global_steps = 0
    start = time.time()
    
    test_acc_list = []
    
    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()
        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        test_acc_list.append(val_acc)
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), "quantized_net_after_finetune.pt")
    return test_acc_list  # my addition, originally doesn't return anything


def test(net):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    return val_acc  # my addition. Originally returns None

