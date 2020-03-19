import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

# Check for a GPU\n"
if not tf.test.gpu_device_name():
    # warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
    print("No GPU")
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



with tf.device("/device:CPU:0"):



    data = keras.datasets.fashion_mnist
    (train_images, train_labels),(test_images, test_labels) = data.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images/255
    test_images = test_images/255


    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    #
    # print("Tested ACC:", test_acc)

    prediction = model.predict(test_images)

    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
        plt.show()



