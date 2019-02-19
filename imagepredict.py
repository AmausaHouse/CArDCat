import glob
import numpy as np

import cv2

import matplotlib.pyplot as plt 

import predict
import facetrimming
import getcontours

color0 = (60, 60, 60)
color1 = (180, 180, 180)

def predict_show(path):
    im = cv2.imread(path)
    rect = facetrimming.get_rect_from_ndarr(im)

    pr = predict.Predict()

    for r in rect:
        cv2.rectangle(im, tuple(r[0:2]),tuple(r[0:2]+r[2:4]), color0, thickness=4)
        cv2.rectangle(im, tuple(r[0:2]),tuple(r[0:2]+r[2:4]), color1, thickness=2)

        resized = cv2.resize(im[r[1]:r[1]+r[3],r[0]:r[0]+r[2]], (64, 64))

        pred = pr.predict_from_ndarr(resized.astype('float32'))

        cv2.putText(im, pred, (int(r[0]+0.05*r[2]),int(r[1]+0.15*r[3])), cv2.FONT_HERSHEY_SIMPLEX, r[3] / 256, color0, 4, cv2.LINE_AA, False)
        cv2.putText(im, pred, (int(r[0]+0.05*r[2]),int(r[1]+0.15*r[3])), cv2.FONT_HERSHEY_SIMPLEX, r[3] / 256, color1, 2, cv2.LINE_AA, False)

        print('{} : {}\n'.format(r, pred))

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.pause(.01)

    input()

class ImagePredictor:

    def __init__(self):
        self.pr = predict.Predict()
        self.gc = getcontours.GetContours()

    def predict(self, path):

        im = cv2.imread(path)

        ret_list = []

        # 輪郭取るやつ (64*64の顔部分画像, 元画像からみた短形データ)
        ret = self.gc.face_shape_detector_dlib(im)

        for roi, rect in ret:

            pred = self.pr.predict_from_ndarr(roi.astype('float32'))

            ret_list.append(dict(index=pred, rect=rect))

        return ret_list
