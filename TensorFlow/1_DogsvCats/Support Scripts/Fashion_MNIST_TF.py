from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

if not tf.test.gpu_device_name():
    # warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
    print("No GPU")
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



root_training = "C:\\Users\\Indiana\\Desktop\\CatsVDogs\\TensorFlow\\Kaggle_DvC\\Training"
root_validation = "C:\\Users\\Indiana\\Desktop\\CatsVDogs\\TensorFlow\\Kaggle_DvC\\Validation"

train_cats_dir = os.path.join(root_training, "Cat")
train_dogs_dir =os.path.join(root_training, "Dog")


validation_cats_dir =os.path.join(root_validation, "Cat")
validation_dogs_dir =os.path.join(root_validation, "Dog")



num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

with tf.device('/device:GPU:0'):
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=root_training,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=root_validation,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(2)
    ])

    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])


    model.summary()


    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )
