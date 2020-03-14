import os
import cv2
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

REBUILD_DATA = False

class DogsVCats():
    IMG_SIZE = 50

    PATH = 'C:\\Users\\Indiana\\Desktop\\CatsVDogs\\PetImages'

    CAT = PATH + '\\Cat'
    DOG = PATH + '\\Dog'
    LABELS = {CAT: 0, DOG: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

    def make_training_data(self):
         for label in self.LABELS:
             for f in tqdm(os.listdir(label)):
                 try:
                     path = os.path.join(label, f)
                     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                     img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                     self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                     if label == self.CAT:
                         self.cat_count += 1
                     elif label == self.DOG:
                         self.dog_count += 1
                 except Exception as e:
                    pass

         np.random.shuffle(self.training_data)
         np.save("'C:\\Users\\Indiana\\Desktop\\CatsVDogs\\training_data.npy", self.training_data)
         print("Cats:", self.cat_count)
         print("Dogs:", self.dog_count)


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

        x = torch.rand(50, 50).view(-1, 1, 50, 50)
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 250)
        self.fc2 = nn.Linear(250, 125)
        self.fc3 = nn.Linear(125, 50)
        self.fc4_BN = nn.BatchNorm1d(2)
        self.fc4 = nn.Linear(50, 2)



    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_BN(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            print("To Linear", self._to_linear)
            return x
        else:
            return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.d1(F.relu(self.fc1(x)))
        x = self.d2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4_BN(self.fc4(x))

        # return F.softmax(x, dim=1)
        return F.softmax(x, dim=1)



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("TORCH: Running on GPU")
    else:
        device = torch.device("cpu")
        print("TORCH: Running on CPI")

    if REBUILD_DATA:
        dogsVcats = DogsVCats()
        dogsVcats.make_training_data()

    # training_data = np.load("C:\\Users\\Indiana\\Desktop\\CatsVDogs\\training_data.npy", allow_pickle=True)
    training_data = np.load("C:\\Users\\Indiana\\Desktop\\Machine Learning\\DataSet\\train\\training__TENSOR_data.npy", allow_pickle=True)
    print(len(training_data))
    print(np.shape(training_data))
    print(training_data[1])
    plt.imshow(training_data[1][0])
    print(training_data[1][1])
    # plt.show()


    net = Net()


    X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X = X/255.0

    Y = torch.Tensor([training_data[i][1] for i in range(len(training_data))])

    # Y_data_ind = [np.argmax(Y_data[i]) for i in range(len(Y_data))]
    #
    # Y = (Y_data_ind)

    print("TARGET IND", Y)

    VAL_PCT = .1
    val_size = int(len(X)*VAL_PCT)
    print(val_size)

    train_X = X[:-val_size]
    train_Y = Y[:-val_size]

    test_X = X[-val_size:]
    test_Y = Y[-val_size:]

    BATCH_SIZE = 200

    EPOCHS = 25



    net.to(device)

    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # loss_function = nn.MSELoss()
    loss_function = nn.MSELoss()
    loss_function.to(device)




    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            net.train()
            # print(i, i+BATCH_SIZE)
            batch_x = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_Y[i:i+BATCH_SIZE]

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            net.zero_grad()
            output = net(batch_x)
            loss = loss_function(output, batch_y)
            loss.backward()
            optimizer.step()
        print("Epoch= ", epoch, "Loss= ", loss)
    # print(loss)


correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        net.eval()
        real_class = torch.argmax(test_Y[i].to(device))
        net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy:", round(correct/total, 3))


NN_TRAINED_Model_path = "C:\\Users\\Indiana\\Desktop\\CatsVDogs\\Trained_model"
NN_TRAINED_Model_path += "\\train_cat_dog_model.pt"
torch.save(net.state_dict(), NN_TRAINED_Model_path)