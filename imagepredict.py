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

    def predict(self,path):

        im = cv2.imread(path)
        rect = facetrimming.get_rect_from_ndarr(im)

        ret_list = []

        for r in rect:

            copy = im.copy()[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]

            copy, roi, conv = self.gc.face_shape_detector_dlib(copy)

            if roi is not None :

                white = np.zeros(roi.shape[0:2], dtype=np.uint8)
                cv2.fillConvexPoly(white, conv, 1)

                roi = cv2.bitwise_and(roi, roi, mask=white)

                resized = cv2.resize(roi, (64, 64))

                pred = self.pr.predict_from_ndarr(resized.astype('float32'))

                data = {}
                data['rect'] = r
                data['name'] = pred

                ret_list.append(data)

        return ret_list
