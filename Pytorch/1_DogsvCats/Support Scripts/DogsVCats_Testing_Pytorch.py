import os
import cv2
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets, utils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# If available use GPU memory to load data
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# print("ON DEVICE", device)