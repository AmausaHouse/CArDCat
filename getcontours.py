import hashlib
import datetime
import multiprocessing

import numpy as np
import os

import cv2
import dlib

from imutils import face_utils

from concurrent.futures import ThreadPoolExecutor

import preprocessing

import glob

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

    def make_faces_picture(self, human_id, movie_id):

        save_folder = './images/human_data/' + human_id + '/png/' + movie_id
        movie_path = './images/human_data/' + human_id + '/mov/' + movie_id + '.mov'

        # 動画が存在していない場合だけ動かす
        if os.path.exists(save_folder):
            return

        os.mkdir(save_folder)

        cpu_num = multiprocessing.cpu_count()
        frame_num = cv2.VideoCapture(movie_path).get(cv2.CAP_PROP_FRAME_COUNT)

        def read(frame_mod):
            cap = cv2.VideoCapture(movie_path)

            for time in range(int(frame_num // cpu_num + 1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cpu_num * time + frame_mod)
                ret, frame = cap.read()
                if not ret:
                    break

                # 動画を回す
                # frame = frame.transpose(1, 0, 2)

                # ここで画像加工
                ret = self.face_shape_detector_dlib(frame)

                # 画像と短形
                for tup in ret:

                    roi = tup[0]

                    picture_id = hashlib.md5(str(datetime.datetime.utcnow()).encode("utf-8")).hexdigest()

                # ここでノイズ
                preprocessing.preprocessing(save_folder, picture_id, roi)

            cap.release()

        with ThreadPoolExecutor(max_workers=cpu_num) as executor:
            executor.map(read, range(cpu_num))
