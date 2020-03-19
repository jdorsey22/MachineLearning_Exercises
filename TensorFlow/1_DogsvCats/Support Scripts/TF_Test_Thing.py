import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from scipy import ndimage




REBUILD_DATA = True

class DogsVCats():
    IMG_SIZE = 50

    PATH = 'C:\\Users\\\Indiana\\Desktop\\Machine Learning\\DataSet\\train'
    test_path = 'C:\\Users\\\Indiana\\Desktop\\Machine Learning\\DataSet\\test'
    out_path = 'C:\\Users\\\Indiana\\Desktop\\Machine Learning\\DataSet\\Output\\out_picts'

    CAT = PATH + '\\Cats'
    DOG = PATH + '\\Dogs'
    LABELS = {CAT: 0, DOG: 1}
    training_data = []
    cat_count = 0
    dog_count = 0


    def make_training_data(self):
        with tf.device("/GPU:0"):
            for label in self.LABELS:
                for f in tqdm(os.listdir(label)):
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                        img_rot = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        angle = np.random.randint(-45, 45)
                        img_rot = ndimage.rotate(img_rot, angle)
                        img_rot = cv2.resize(img_rot, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img_rot), np.eye(2)[self.LABELS[label]]])

                        if label == self.CAT:
                            self.cat_count += 2
                        elif label == self.DOG:
                            self.dog_count += 2

                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data)

        print("DATA Length", len(self.training_data))
        np.save( self.out_path + "\\training_TENSOR_data.npy", self.training_data)
        print("Cats:", self.cat_count)
        print("Dogs:", self.dog_count)

    def data_augmentation(self):
        for label in self.LABELS:
            for num, file in enumerate(os.listdir(label)):
                path = os.path.join(label, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                angle = np.random.randint(-45, 45)
                img_rot = ndimage.rotate(img, angle)
                img_rot = cv2.resize(img_rot, (self.IMG_SIZE, self.IMG_SIZE))
                cv2.imwrite(self.out_path + '_' + str(num) + '.png', img_rot)




if __name__ == '__main__':

    if not tf.test.gpu_device_name():
        # warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
        print("No GPU")
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    print(tf.__version__)

    D = DogsVCats()

    if REBUILD_DATA:
        D.make_training_data()