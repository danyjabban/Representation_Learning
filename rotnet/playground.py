#%%
import sys
sys.path.append('../')
from data_utils import *
#%%

import cv2
import matplotlib.pyplot as plt
breakpoint()
rot_DL = CIFAR10_RotNet()
rot_DL[0]