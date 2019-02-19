import os
import numpy as np

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model

import getdata

epoch = 15
max_match = 100000
max_not_match = 100000

def cnn(ids):

    label_size, data, label = getdata.make_dataset(ids)

    model = Sequential()
    model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(2, 2), strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(label_size + 1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # result = model.fit(data, label, epochs=epoch, batch_size=32, validation_data=(testdata, testlabel))

    result = model.fit(data, label, epochs=epoch, batch_size=32, validation_split=0.1)

    fig = plt.figure(1)
    for label_name in list(filter(lambda x:x.find('loss')!=-1, result.history.keys())):
        plt.plot(range(1, epoch+1), result.history[label_name], label=label_name)
    plt.xlabel('epochs')
    plt.legend()
    fig.savefig('./models/output_loss.png')

    fig = plt.figure(2)
    for label_name in list(filter(lambda x:x.find('acc')!=-1, result.history.keys())):
        plt.plot(range(1, epoch+1), result.history[label_name], label=label_name)
    plt.xlabel('epochs')
    plt.legend()
    fig.savefig('./models/output_acc.png')

    json = model.to_json()
    open('./models/model.json', 'w').write(json)
    model.save_weights('./models/model_weight.hdf5')
