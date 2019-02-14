import os

import matplotlib
matplotlib.use('Agg')

from keras.models import model_from_json
from keras.layers import *

import const

vfunc = np.vectorize(lambda x: x / 255.0)

class Predict:

    def __init__(self):

        modelpath = os.path.join(os.path.dirname(__file__), 'models/model.json')
        weightpath = os.path.join(os.path.dirname(__file__), 'models/model_weight.hdf5')

        # cnnのあれ
        self.model = model_from_json(open(modelpath, 'r').read())
        self.model.load_weights(weightpath)

        self.names = []

        for name, movies in const.name_movies:
            self.names.append(name)

        self.names.append('not match')

    def predict_from_ndarr(self, im):
        im = vfunc(im)
        data = np.array([im])

        return self.names[np.argmax(self.model.predict(data)[0])]
