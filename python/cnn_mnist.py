#!/usr/bin/python3

import sys
from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def load_dataset():
    """
    Load local MNIST dataset.
    :return:    train_images, train_labels, test_images, test_labels
    """
    mnist_dir = "../data/mnist/"
    x_train, y_train = loadlocal_mnist(images_path='{}train-images-idx3-ubyte'.format(mnist_dir),
                                       labels_path='{}train-labels-idx1-ubyte'.format(mnist_dir))
    x_test, y_test = loadlocal_mnist(images_path='{}t10k-images-idx3-ubyte'.format(mnist_dir),
                                     labels_path='{}t10k-labels-idx1-ubyte'.format(mnist_dir))
    return x_train.reshape((60000, 28, 28)), y_train, x_test.reshape((10000, 28, 28)), y_test


def show_first_images(images):
    """
    Plot first 9 images
    :param      images
    """
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(images[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()  # show the figure


def reshape_and_normalise(images):
    """
    Reshaping the array to 4-dims so that it can work with the Keras API.
    And normalise so the values are between 1 and 0.
    :param      images
    :return:    images:     shape(_28,28,1)
    """
    # Reshaping the array to 4-dims so that it can work with the Keras API
    images = images.reshape(images.shape[0], 28, 28, 1)
    images = images.astype('float32')  # Making sure that the values are float
    return images / 255  # Normalise so that values are between 0 and 1


def define_model():
    """
    Creating a Sequential Model and adding the layers
    """
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    return model


def main(argv):
    train_images, train_labels, test_images, test_labels = load_dataset()
    train_images = reshape_and_normalise(train_images)
    test_images = reshape_and_normalise(test_images)

    #show_first_images(train_images)

    model = define_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=train_images, y=train_labels, epochs=10)
    model.evaluate(test_images, test_labels)


if __name__ == "__main__":
    main(sys.argv)