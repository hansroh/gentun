#!/usr/bin/env python
"""
Test the GeneticCnnModel using the MNIST dataset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnModel
    from keras.datasets import cifar10
    '''
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))

    selection = random.sample(range(n), 10000)
    y_train = lb.transform(train_labels[selection])
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data
    '''
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    n = train_images.shape[0]
    test_n = test_images.shape[0]
    input_shape = train_images[0].shape
    lb = LabelBinarizer()
    lb.fit(range(10))
    # print("Here")
    # selection = random.sample(range(n), 100000)  # Use only a subsample
    # test_selection = random.sample(range(test_n), 10000)
    # print("Here")
    # y_train = lb.transform(train_labels[selection])  # One-hot encodings
    y_train = lb.transform(train_labels)  # One-hot encodings
    # print("Here")
    # test_sel_labels = test_labels[test_selection]
    test_sel_labels = test_labels
    y_test = lb.transform(test_sel_labels)
    # print("Here")
    if len(train_images.shape) < 4:
        new_shape = (*train_images.shape, 1)
        new_shape_test = (*test_images.shape, 1)
    else:
        new_shape = train_images.shape
        new_shape_test = test_images.shape

    x_train = train_images.reshape(new_shape)
    x_test = test_images.reshape(new_shape_test)
    x_train = x_train / 255  # Normalize train data
    x_test = x_test / 255
    model = GeneticCnnModel(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        genes={'S_1': '000', 'S_2': '000000', 'S_3': '0000000000'},  # Genes to test
        nodes=(3, 4, 5),  # Number of nodes per DAG (corresponds to gene bytes)
        input_shape=input_shape,  # Shape of input data
        kernels_per_layer=(20, 50, 50),  # Number of kernels per layer
        kernel_sizes=((5, 5), (5, 5), (5, 5)),  # Sizes of kernels per layer
        dense_units=500,  # Number of units in Dense layer
        dropout_probability=0.5,  # Dropout probability
        classes=10,  # Number of classes to predict
        kfold=5,
        epochs=1,
        learning_rate=1e-5,
        batch_size=1024, gpu="1"
    )
    print(model.validate())
