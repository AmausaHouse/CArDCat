import os

import matplotlib
matplotlib.use('Agg')

from keras.models import model_from_json
from keras.layers import *

vfunc = np.vectorize(lambda x: x / 255.0)

class Predict:

    def __init__(self):

        modelpath = os.path.join(os.path.dirname(__file__), 'models/model.json')
        weightpath = os.path.join(os.path.dirname(__file__), 'models/model_weight.hdf5')

        # cnnのあれ
        self.model = model_from_json(open(modelpath, 'r').read())
        self.model._make_predict_function()
        self.model.load_weights(weightpath)

    # データベースのindexと一致する 名前の数がNとして、照合しないならNを返す(ソースコードを読め)
    def predict_from_ndarr(self, im):
        im = vfunc(im)
        data = np.array([im])

        return np.argmax(self.model.predict(data)[0])
