import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os











#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# train_features, train_labels = extract_features(train_dir, train_size)  # Agree with our small dataset size
# validation_features, validation_labels = extract_features(validation_dir, validation_size)
# test_features, test_labels = extract_features(test_dir, test_size)
#
# from keras import models
# from keras import layers
# from keras import optimizers
#
# epochs = 100
#
# model = models.Sequential()
# model.add(layers.Flatten(input_shape=(7, 7, 512)))
# model.add(layers.Dense(256, activation='relu', input_dim=(7 * 7 * 512)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()
#
# # Compile model
# model.compile(optimizer=optimizers.Adam(),
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# # Train model
# history = model.fit(train_features, train_labels,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_data=(validation_features, validation_labels))
#
#
#
#
#
#



REBUILD_DATA = True
CREATE_TEST_DATA = False



class MakeData():
    IMG_SIZE = 50

    PATH = 'C:\\Users\\Indiana\\Desktop\\CatsVDogs\\PetImages'
    output_path = 'C:\\Users\\Indiana\\Desktop\\CatsVDogs\\TensorFlow'


    CAT = PATH + '\\Cat'
    DOG = PATH + '\\Dog'
    LABELS = {CAT: 0, DOG: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

    test_pct = .1 # 10% of data set to be set aside for out of sample testing.
    raw_data = []
    data = []
    labels = []
    test_data = []
    train_data = []
    test_label = []
    train_label = []

    def make_training_data(self):
         for label in self.LABELS:
             for f in tqdm(os.listdir(label)):
                 try:
                     path = os.path.join(label, f)
                     img = cv2.imread(path, cv2.IMREAD_COLOR)
                     img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                     self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                     cv2.imshow("Window", img)
                     cv2.waitKey(100)

                     if label == self.CAT:
                         self.cat_count += 1
                     elif label == self.DOG:
                         self.dog_count += 1
                 except Exception as e:
                    pass

         np.random.shuffle(self.training_data)
         np.save(self.output_path + "\\raw_data_TF_COLOR.npy", self.training_data)
         print("Cats:", self.cat_count)
         print("Dogs:", self.dog_count)

    def create_test_data(self):
        self.raw_data = np.load(self.output_path + "\\raw_data_TF.npy", allow_pickle=True)
        test_data_len = int(self.test_pct * len(self.raw_data))

        for features, label in self.raw_data:
            self.data.append(features)
            self.labels.append(label)

        self.data = np.array(self.data).reshape(-1, self.IMG_SIZE, self.IMG_SIZE)



        self.test_data = [self.data[i] for i in range(0, test_data_len)]
        self.test_label = [self.labels[i] for i in range(0, test_data_len)]
        self.train_data = [self.data[i] for i in range(test_data_len, len(self.raw_data))]
        self.train_label = [self.labels[i] for i in range(test_data_len, len(self.raw_data))]

        np.save(self.output_path + "\\training_data_TF.npy", self.train_data)
        np.save(self.output_path + "\\training_label_TF.npy", self.train_label)
        np.save(self.output_path + "\\testing_label_TF.npy", self.test_label)
        np.save(self.output_path + "\\testing_data_TF.npy", self.test_data)


        print(np.shape(self.test_data[0]))
        # print("Test Data:", self.test_data )
        # print("Test Label:", self.test_label)




if __name__ == '__main__':

    dogsVcats = MakeData()
    output_path = 'C:\\Users\\Indiana\\Desktop\\CatsVDogs\\TensorFlow'


    if REBUILD_DATA:
        print("1")
        dogsVcats.make_training_data()
    if CREATE_TEST_DATA:
        print("2")
        dogsVcats.create_test_data()


    train_images = np.load(output_path + "\\training_data_TF.npy", allow_pickle=True)
    train_labels = np.load(output_path + "\\training_label_TF.npy", allow_pickle=True)
    test_images = np.load(output_path + "\\testing_data_TF.npy", allow_pickle=True)
    test_labels = np.load(output_path + "\\testing_label_TF.npy", allow_pickle=True)

    # train_images = np.expand_dims(train_images, axis=0)

    train_images = train_images.reshape(-1, 50, 50, 1)
    test_images = test_images.reshape(-1, 50, 50, 1)


    class_names = ['Cat', 'Dog']

    train_images = train_images/255
    test_images = test_images/255

    # for i in range(10):
    #     plt.imshow(train_images[i])
    #     plt.show()

    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), input_shape=(50, 50, 1), activation='relu'),
      tf.keras.layers.MaxPool2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPool2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPool2D((2, 2)),
      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(64, activation='relu'),
      # tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(150, activation='relu'),
      # tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    #
    print("Tested ACC:", test_acc)

    # prediction = model.predict(test_images)
    #
    # for i in range(5):
    #     plt.grid(False)
    #     plt.imshow(test_images[i], cmap=plt.cm.binary)
    #     plt.xlabel("Actual: " + class_names[test_labels[i]])
    #     plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    #     plt.show()
