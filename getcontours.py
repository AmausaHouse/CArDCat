import hashlib
import datetime

import numpy as np

import cv2
import dlib

import imutils
from imutils import face_utils

import matplotlib.pyplot as plt

import const

predictor = dlib.shape_predictor(const.predictor_path)
detector = dlib.get_frontal_face_detector()

def face_shape_detector_dlib(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets, scores, idx = detector.run(img_rgb, 0)

    if len(dets) > 0:
        for rect in dets:
            shape = predictor(img_rgb, rect)
            shape = face_utils.shape_to_np(shape)
            copy = img.copy()

            conv = cv2.convexHull(shape)

            (x, y, w, h) = cv2.boundingRect(np.array([shape[0:68]])) 
            roi = img[y:y+h, x:x+w]

            for i in range(conv.shape[0]):
                (px,py) = conv[i][0]
                cv2.putText(copy, str(i), (px + 10, py + 10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(copy, (px, py), 1, (0, 0, 255), 10)

            for d2 in conv:
                for d1 in d2:
                    d1[0] = int((d1[0] - x) * 64 / h)
                    d1[1] = int((d1[1] - y) * 64 / w)

            if roi.shape[0:2].count(0):
                return copy, None, None

            roi = cv2.resize(roi,(64,64))

        return copy, roi, conv
    else :
        return img, None, None

def make_faces_picture(human_name, movie_path):

    save_folder = './images/human_data/faces/' + human_name + '/'

    cap = cv2.VideoCapture(movie_path)
    ret = True

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = frame.transpose(1, 0, 2)
        frame, roi, conv = face_shape_detector_dlib(frame)

        if const.show_contours:
            cv2.imshow('img', frame)

        if roi is not None :

            white = np.zeros(roi.shape[0:2], dtype=np.uint8)
            cv2.fillConvexPoly(white, conv, 1)

            roi = cv2.bitwise_and(roi, roi, mask=white)

            if const.show_contours:
                plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            name = hashlib.md5(str(datetime.datetime.utcnow()).encode("utf-8")).hexdigest()
            cv2.imwrite(save_folder + name + '.png', roi)

        cv2.waitKey(1)
        plt.pause(.01)

    cap.release()
    cv2.destroyAllWindows()
