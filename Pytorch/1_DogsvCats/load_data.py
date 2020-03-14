import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# If available use GPU memory to load data
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("ON DEVICE", device)


class CatsAndDogsDataset(Dataset):
    """ Dogs & Cats data Set """
    def __init__(self, REBUILD_DATA=False, IMG_SIZE=50, transform=None):
        """ This function, inputs REBUILD Flag. and """
        self.IMG_SIZE = IMG_SIZE
        self.transform = transform

        self.PATH = "C:\\Users\\jonat\\Desktop\\Machine Learning\\DataSet\\train"

        self.CAT = self.PATH + '\\Cats'
        self.DOG = self.PATH + '\\Dogs'
        self.LABELS = {self.CAT: 0, self.DOG: 1}
        self.REBUILD_DATA = REBUILD_DATA

        self.data = []

        if self.REBUILD_DATA:

            for label in self.LABELS:
                for f in tqdm(os.listdir(label)):
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.data.append([torch.from_numpy(np.array(img)), np.eye(2)[self.LABELS[label]]])

                    except Exception as e:
                        pass

            np.random.shuffle(self.data)
            np.save("C:\\Users\\jonat\\Desktop\\Machine Learning\\DataSet\\data.npy", self.data)

        else:

            self.data = np.load("C:\\Users\\jonat\\Desktop\\Machine Learning\\DataSet\\data.npy", allow_pickle=True)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item]


def random_split(total_data_set, validation_percentage):
    data_set_len = len(total_data_set)
    val_set_pct = float(validation_percentage/100.0)
    val_set_len = int(data_set_len * val_set_pct)
    train_set_len = (data_set_len - val_set_len)

    # total_data_set = np.random.shuffle(total_data_set)

    training_data = total_data_set[0:train_set_len]
    valid_data = total_data_set[(train_set_len+1):]

    return training_data, valid_data


if __name__ == '__main__':
    # creating the dataset object
    dataset = CatsAndDogsDataset(REBUILD_DATA=False)
    # Randomly split dataset into trainset and the validation set
    train_data, val_data = random_split(dataset, validation_percentage=10)

data_set = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

EPOCH = 5

for epoch in range(EPOCH):
    print("EPOCH", epoch)
    for i, (imgs, labels) in enumerate(dataset):
        print("Iteration:", i, "Image", imgs, "Label:", labels)

        if i >= 4:
            break

