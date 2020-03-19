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



#"C:\\Users\\jonat\\Desktop\\Machine Learning\\DataSet\\train"


class CatsAndDogsDataset(Dataset):
    """ Dogs & Cats data Set """
    def __init__(self, REBUILD_DATA=False, IMG_SIZE=100, transform=None, root="C:\\Users\\Indiana\\Desktop\\Machine Learning\\DataSet\\train_extended" ):

        """ This function, inputs REBUILD Flag. and """
        self.IMG_SIZE = IMG_SIZE
        self.transform = transform

        self.PATH = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item]


def random_split(total_data_set, validation_percentage):
    data_set_len = len(total_data_set)
    val_set_pct = float(validation_percentage/100.0)
    val_set_len = int(data_set_len * val_set_pct)
    train_set_len = (data_set_len - val_set_len)

    # training_data = total_data_set[0:train_set_len]
    # valid_data = total_data_set[(train_set_len+1):]

    training_data = [total_data_set[i] for i in range(train_set_len)]
    valid_data = [total_data_set[i] for i in range(train_set_len, data_set_len)]

    return training_data, valid_data



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv2_BN = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 200, 5)

        self.d1 = nn.Dropout(p=.2)
        self.d2 = nn.Dropout(p=.2)

        self._to_linear = None

        x = torch.rand(100, 100).view(-1, 1, 100, 100)
        self.convs(x)

        # self.fc1 = nn.Linear(self._to_linear, 250)
        # self.fc2 = nn.Linear(250, 400)
        # self.fc3 = nn.Linear(400, 200)
        # self.fc4_BN = nn.BatchNorm1d(50)
        # self.fc4 = nn.Linear(200, 50)
        # self.fc5 = nn.Linear(50, 2)

        # self.fc1 = nn.Linear(self._to_linear, 400)
        # self.fc2 = nn.Linear(400, 100)
        # self.fc3 = nn.Linear(100, 2)

        self.fc1 = nn.Linear(self._to_linear, 400)
        self.fc2 = nn.Linear(400, 2)





    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_BN(self.conv2(x))), (4, 4))
        x = F.max_pool2d(F.relu(self.conv3(x)), (3, 3))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            print("To Linear", self._to_linear)
            return x
        else:
            return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        # x = self.d1(F.relu(self.fc1(x)))
        # x = self.d2(F.relu(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        # x = self.fc4_BN(self.fc4(x))
        # x = self.fc5(x)

        # x = self.d1(F.relu(self.fc1(x)))
        # x = self.d2(F.relu(self.fc2(x)))
        # x = self.fc3(x)

        x = self.d1(F.relu(self.fc1(x)))
        x = self.fc2(x)


        return F.softmax(x, dim=1)



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("TORCH: Running on GPU")
    else:
        device = torch.device("cpu")
        print("TORCH: Running on CPU")

    data_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    # creating the dataset object
    # dataset = CatsAndDogsDataset(REBUILD_DATA=False, transform=data_transform)
    dataset = datasets.ImageFolder(root='C:\\Users\\Indiana\\Desktop\\Machine Learning\\DataSet\\train_extended', transform=data_transform)


    # Randomly split dataset into trainset and the validation set
    train_data, val_data = random_split(dataset, validation_percentage=20)

    # Data Setup
    train_data = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=3)
    val_data = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=3)



    # Model Setup
    net = Net()

    EPOCHS = 5

    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # loss_function = nn.MSELoss()
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)

    for epoch in range(EPOCHS):
        print("EPOCH", epoch)
        for (imgs, labels) in tqdm(train_data):
            net.train()
            net.zero_grad()
            output = net(imgs.float().view(-1, 1, 100, 100).to(device))
            # loss = loss_function(output, labels.float().to(device))
            loss = loss_function(output, torch.argmax(labels, dim=1).to(device))



            loss.backward()
            optimizer.step()
        print("Epoch= ", epoch, "Loss= ", loss)

        with torch.no_grad():
            correct = 0
            total = 0
            for (imgs, labels) in tqdm(val_data):
                net.eval()
                real_class = torch.argmax(labels.to(device))
                net_out = net(imgs.float().view(-1, 1, 100, 100).to(device))[0]
                predicted_class = torch.argmax(net_out)
                if predicted_class == real_class:
                    correct += 1
                total += 1
        print("Validation Accuracy:", round(correct/total, 3))

        # train_data = DataLoader(train_data, batch_size=200, shuffle=True, num_workers=3)
        # val_data = DataLoader(val_data, batch_size=200, shuffle=True, num_workers=3)


    NN_TRAINED_Model_path = "C:\\Users\\Indiana\\Documents\\MachineLearning_Exercises\\Pytorch\\1_DogsvCats\\NN_Model"
    NN_TRAINED_Model_path += "\\Dogs_V_Cats_NN_model.pt"
    torch.save(net.state_dict(), NN_TRAINED_Model_path)