import os
import numpy as np

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.utils import plot_model

import const
import getdata

vfunc = np.vectorize(lambda x: x / 255.0)

class Predict:

    def __init__(self):

        self.model = model_from_json(open('./models/model.json', 'r').read())
        self.model.load_weights('./models/model_weight.hdf5')

        self.names = []

        for name, movies in const.name_movies:
            self.names.append(name)

        self.names.append('not match')

    def predict_from_ndarr(self, im):
        im = vfunc(im)
        data = np.array([im])

        return self.names[np.argmax(self.model.predict(data)[0])]


    def predict_from_path(self, path):
        im = cv2.imread(path).astype('float32')

        return self.predict_from_image(im)