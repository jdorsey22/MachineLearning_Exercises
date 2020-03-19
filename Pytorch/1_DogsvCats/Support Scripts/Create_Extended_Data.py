import os
import cv2
import numpy as np
from scipy import ndimage

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

PATH = "C:\\Users\\Indiana\\Desktop\\Machine Learning\\DataSet\\train_extended"
Labels = ["\\Cats", "\\Dogs"]
Labels_Ext = ["\\Cats_Extended", "\\Dogs_Extended"]

SUBDIR = [PATH + Labels[i] for i in range(len(Labels))]
OutPath = [PATH + Labels_Ext[i] for i in range(len(Labels_Ext))]

counter = 0
flip_counter = 0

for subdir_path in SUBDIR:
    for f in os.listdir(subdir_path):

        FILE_NAME = subdir_path + "\\" + f
        img = cv2.imread(FILE_NAME)

        # height, width, number of channels in image
        height, width, channels = img.shape


        img_ext = img

        angle = np.random.randint(-45, 45)
        img_ext = ndimage.rotate(img_ext, angle)
        img_ext = cv2.resize(img_ext, (height, width))

        if flip_counter % 5 == 0:
            img_flip = cv2.flip(img, 1)
            img_flip_ext = cv2.flip(img_ext, 1)

            img = img_flip
            img_ext = img_flip_ext

        for out in OutPath:

            cv2.imwrite(out + "\\" + str(counter) + "_.jpg", img_ext)
            cv2.imwrite(out + "\\" + str(counter + 1) + "_.jpg", img)

            counter += 2


for thing in OutPath:
    length = len(os.listdir(thing))
    print("Number in DataSet:",length )



