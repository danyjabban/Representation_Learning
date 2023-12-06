"""
Dataset Utils
"""

import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset 
import random


class CIFAR10_SimCLR(CIFAR10):
    """
    CIFAR Dataset object for simCLR data
    """
    def __init__(self, root='data', train=True):
        super().__init__(root, train)
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
                
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            transforms.RandomResizedCrop(size=32, antialias=True),#, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(s=0.5),
          ]
        )
        
        im1_trans = transform_train(img)
        im2_trans = transform_train(img)
        return im1_trans, im2_trans, target



class CIFAR10_RotNet(CIFAR10):
    """
    CIFAR Dataset object for RotNet data
    """
    def __init__(self, root='data', train=True):
        super().__init__(root, train)
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        rot_ims = []
        for rot in [0, 90, 180, 270]:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation((rot,rot)),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
            
            im_rot = transform_train(img)
            rot_ims.append(im_rot.permute(1,2,0))
        rot_ims = torch.stack(rot_ims)
        
        return rot_ims, torch.tensor([1,2,3,4])
    
    
    
class CIFAR10_train(CIFAR10):
    """
    CIFAR Dataset object for finetuning
    """
    def __init__(self, root='data', train=True):
        super().__init__(root, train)
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=32, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5)
            #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ])
        img = transform_train(img)
        return img, target
    

class CIFAR10_test(CIFAR10):
    """
    CIFAR Dataset object for finetuning
    """
    def __init__(self, root='data', train=False):
        super().__init__(root, train)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        transform_test = transforms.Compose([
            transforms.ToTensor()])
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
            
        img = transform_test(img)
        return img, target
    


def get_color_distortion(s):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort



def get_indices(p, ds, seed=661):
    if seed:
        random.seed(seed)
    
    classes = dict()
    for i in range(len(ds)):
        cl = ds[i][1]
        classes[cl] = classes.get(cl, []) + [i]
        
    class_list = []
    for k in classes.keys():
        random.shuffle(classes[k])
        class_list = class_list + classes[k][:int(len(classes[k])*p)]
        
    return class_list


def write_indices(p, fname_train, base_path):
    """
    p: fraction of train data that's included in finetuning
    """
    trainset_dummy = CIFAR10_train(train=True)
    os.makedirs(base_path, exist_ok=True)
    if fname_train not in set(os.listdir(base_path)):
        fptr = open(base_path + fname_train, 'w')
        train_idx = get_indices(p, ds=trainset_dummy)
        print(len(train_idx), p)
        for _, idx in enumerate(train_idx):
            fptr.write("%d\n" % idx)
        fptr.close()
    else:
        print(base_path + fname_train, "already exists")
    return


def get_train_test_sets(fname_train, base_path):
    def read_indices(fname, base_path):
        fptr = open(base_path + fname, 'r')
        lines = fptr.readlines()
        fptr.close()
        ret = []
        for i in lines:
            ret.append(int(i))
        return ret
    train_set = Subset(CIFAR10_train(train=True), read_indices(fname_train, base_path))
    # ^^ uses train data and train transformation

    test_set = CIFAR10_test(train=False)
    return train_set, test_set
