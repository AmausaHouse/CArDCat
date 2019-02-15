import hashlib
import datetime

import numpy as np

import cv2
import dlib

from imutils import face_utils

import glob

import matplotlib.pyplot as plt

import const

class GetContours:

    def __init__(self):
        self.predictor = dlib.shape_predictor(const.predictor_path)
        self.detector = dlib.get_frontal_face_detector()

    # 元画像を渡されてから顔部分を検出して64*64に変換, 凸包とtupleで投げる
    def face_shape_detector_dlib(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 顔検出
        dets = self.detector.run(img_rgb, 0)[0]

        tuplis = []

        # 顔部分の取得
        for rect in dets:
            copy = img.copy()

            # 顔パーツの検出
            shape = self.predictor(img_rgb, rect)
            shape = face_utils.shape_to_np(shape)

            # 検出した顔パーツから凸包
            conv = cv2.convexHull(shape)

            # 黒画像を生成
            white = np.zeros(copy.shape[0:2], dtype=np.uint8)
            # 凸包内部を白にする
            cv2.fillConvexPoly(white, conv, color=255)

            # 凸包外部を黒にした画像を生成
            copy = cv2.bitwise_and(copy, copy, mask=white)

            # 顔部分の切り出し
            (x, y, w, h) = cv2.boundingRect(np.array([shape[0:68]]))
            roi = copy[y:y+h, x:x+w]
            if roi.shape[0] < 64 or roi.shape[1] < 64:
                continue

            tuplis.append((cv2.resize(roi, (64, 64)), rect))

        return tuplis

    """
    loadpath = './images/learning_data/hoge/'
    savepath = './images/learning_data/huga/'
    """
    def get_dataset(self, loadpath, savepath):
        pathes = glob.glob(loadpath + '*.jpg')
        for p in pathes:
            img = cv2.imread(p)

            ret = self.face_shape_detector_dlib(img)

            # 画像と短形
            for tup in ret:

                roi = tup[0]

                name = hashlib.md5(str(datetime.datetime.utcnow()).encode("utf-8")).hexdigest()
                cv2.imwrite(savepath + name + '.png', roi)



    def make_faces_picture(self, human_name, movie_path):

        save_folder = './images/human_data/faces/' + human_name + '/'

        cap = cv2.VideoCapture(movie_path)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # 動画を回す
            frame = frame.transpose(1, 0, 2)

            ret = self.face_shape_detector_dlib(frame)

            """
            if const.show_contours:
                cv2.imshow('img', frame)
            """

            # 画像と短形
            for tup in ret:

                roi = tup[0]

                """
                if const.show_contours:
                    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                """

                name = hashlib.md5(str(datetime.datetime.utcnow()).encode("utf-8")).hexdigest()
                cv2.imwrite(save_folder + name + '.png', roi)

            """
            cv2.waitKey(1)
            plt.pause(.01)
            """

        cap.release()
        cv2.destroyAllWindows()
