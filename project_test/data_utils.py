"""
Dataset Utils
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset 


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
            transforms.RandomResizedCrop(size=32),#, scale=(0.08, 1.0), antialias=True),
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
            transforms.RandomResizedCrop(size=32, scale=(.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
            
            
        img = transform_train(img)
        return img.permute(1,2,0), target
    

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
        return img.permute(1,2,0), target
    


def get_color_distortion(s):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort



def get_indices(p, ds):
    # torch.manual_seed(0)
    # num_inds = round(p * l)
    # inds = torch.randperm(l)[:num_inds]
    # return inds
    
    classes = dict()
    for i in range(len(ds)):
        cl = ds[i][1]
        classes[cl] = classes.get(cl, []) + [i]
        
    class_list = []
    for k in classes.keys():
        class_list = class_list + classes[k][:int(len(classes[k])*p)]
        
    return class_list



