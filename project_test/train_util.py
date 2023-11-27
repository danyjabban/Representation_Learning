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

from resnet import ResNetCIFAR

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


def nt_xent_loss(x, temperature=0.1):
    # xcs = F.cosine_similarity(x[None, :, :].to('cpu'), x[:, None, :].to('cpu'), dim=-1)
    xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
    # assume x has shape [batch_size*2, 64]
    xcs[torch.eye(x.size(0)).bool()] = float("-inf")

    target = torch.arange(x.size(0))
    target[0::2] += 1
    target[1::2] -= 1

    # ce_loss = F.cross_entropy(xcs / torch.tensor(temperature).to('cpu'), target.to('cpu'), reduction="mean")
    ce_loss = F.cross_entropy(xcs.to(device) / torch.tensor(temperature).to(device), target.to(device), reduction="mean")
    # Standard cross entropy loss
    # only need adjacent pairs: [2k, 2k-1] and [2k-1, 2k] for k in range(batch_size/2),
    # which is ensured by "target" (some magic happens I guess)
    return ce_loss


def augment_data(batch):
    augment = transforms.Compose([
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),  # extra vertical flip?
                                  torchvision.transforms.ColorJitter(0.5)
                                 ])
    new_batch = torch.cat((torch.zeros(batch.shape), torch.zeros(batch.shape)), dim=0)
    # new_batch should be size [batch_size*2, 1, 32, 32]
    new_batch[::2, :, :, :] = augment(batch)
    new_batch[1::2, :, :, :] = augment(batch)
    del batch
    return new_batch


def train(rank, world_size, epochs, batch_size, lr, reg, head, log_every_n=50):
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

    orig_batch_size = batch_size
    every_nth_minibatch = int(1)
    
    # if orig_batch_size > 2048:  # batch size too large, 
    #     # see this link https://discuss.pytorch.org/t/how-can-you-train-your-model-on-large-batches-when-your-gpu-can-t-hold-more-than-a-few-samples/80581/4
    #     batch_size = 2048  # this is max "batch_size" that's actually allowed. orig_batch_size > 2048 is 
    #     # simulated using 1x update for the optimiser per multiple batches (of size 2048). 
    #     # every_nth_minibatch = 2048: this is to enable training using large batch size without out-of-memory
    #     every_nth_minibatch = int(orig_batch_size / batch_size)
    assert orig_batch_size % batch_size == 0

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # criterion = nn.CrossEntropyLoss()
    criterion = nt_xent_loss

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    optimizer = LARS(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=10, verbose=True)
    scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
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
            augment_inputs = augment_data(inputs)
            del inputs
            augment_inputs = augment_inputs.to(rank)  # do not targets.to(device)
            optimizer.zero_grad()
            outputs = net(augment_inputs)
            # outputs = checkpoint(net, augment_inputs)
            del augment_inputs
            # loss = criterion(outputs, targets)
            loss = criterion(outputs, rank)
            loss.backward()
            optimizer.step()
            # if batch_size < orig_batch_size:  # if batch size is very very large, 
            #     # update every batch size.
            #     if batch_idx % every_nth_minibatch == 0:
            #         optimizer.step()
            # else:
            #     optimizer.step()
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
                augment_inputs = augment_data(inputs)
                augment_inputs = augment_inputs.to(rank)  # do not targets.to(device)
                outputs = net(augment_inputs)
                # loss = criterion(outputs, targets)
                loss = criterion(outputs, rank)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
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



def train_no_DDP(net, epochs, batch_size, lr, reg, log_every_n=50):
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
    best_loss = 1000000  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    # orig_batch_size = batch_size
    # every_nth_minibatch = int(1)
    
    # if orig_batch_size > 2048:  # batch size too large, 
    #     # see this link https://discuss.pytorch.org/t/how-can-you-train-your-model-on-large-batches-when-your-gpu-can-t-hold-more-than-a-few-samples/80581/4
    #     batch_size = 2048  # this is max "batch_size" that's actually allowed. orig_batch_size > 2048 is 
    #     # simulated using 1x update for the optimiser per multiple batches (of size 2048). 
    #     # every_nth_minibatch = 2048: this is to enable training using large batch size without out-of-memory
    #     every_nth_minibatch = int(orig_batch_size / batch_size)
    # assert orig_batch_size % batch_size == 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # criterion = nn.CrossEntropyLoss()
    criterion = nt_xent_loss

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    optimizer = LARS(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=10, verbose=True)
    scheduler_after  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
    # T_0 = epochs -> no restart? Paper uses lin warmup (first 10 epochs) then cosine decay schedule w/o restarts

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        if epoch >= 10:
            scheduler = scheduler_after
        else:
            scheduler = scheduler_warmup
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        # correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            augment_inputs = augment_data(inputs)
            del inputs
            augment_inputs, targets = augment_inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(augment_inputs)
            del augment_inputs
            # loss = criterion(outputs, targets)
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()
            # if batch_size < orig_batch_size:  # if batch size is very very large, 
            #     # update every batch size.
            #     if batch_idx % every_nth_minibatch == 0:
            #         optimizer.step()
            # else:
            #     optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
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
                augment_inputs = augment_data(inputs)
                augment_inputs, targets = augment_inputs.to(device), targets.to(device)
                outputs = net(augment_inputs)
                # loss = criterion(outputs, targets)
                loss = criterion(outputs)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_loss = test_loss / num_val_steps
        # val_acc = correct / total
        # print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))
        print("Test Loss=%.4f" % val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            if epoch % 100 == 0:
                print("Saving...")
                torch.save(net.state_dict(), "test_model_epoch_%d.pt" % int(epoch))
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

