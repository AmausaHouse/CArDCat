import cv2

import matplotlib.pyplot as plt

import predict
import facetrimming

color0 = (60, 60, 60)
color1 = (180, 180, 180)

def videopredict():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)

    ret, im = cap.read()

    pr = predict.Predict()

    while ret:
        ret, im = cap.read()

        rect = facetrimming.get_rect_from_ndarr(im)


        for r in rect:
            cv2.rectangle(im, tuple(r[0:2]),tuple(r[0:2]+r[2:4]), color0, thickness=4)
            cv2.rectangle(im, tuple(r[0:2]),tuple(r[0:2]+r[2:4]), color1, thickness=2)

            resized = cv2.resize(im[r[1]:r[1]+r[3],r[0]:r[0]+r[2]], (64, 64))

            pred = pr.predict_from_ndarr(resized)

            cv2.putText(im, pred, (int(r[0]+0.05*r[2]),int(r[1]+0.15*r[3])), cv2.FONT_HERSHEY_SIMPLEX, r[3] / 256, color0, 4, cv2.LINE_AA, False)
            cv2.putText(im, pred, (int(r[0]+0.05*r[2]),int(r[1]+0.15*r[3])), cv2.FONT_HERSHEY_SIMPLEX, r[3] / 256, color1, 2, cv2.LINE_AA, False)

            # print('{} : {}\n'.format(r, pred))

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.plot()
        plt.pause(.01)

videopredict()
