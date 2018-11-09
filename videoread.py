import numpy as np
import hashlib
import datetime

import cv2

import facetrimming

def videoread(human_name, path):

    save_folder = './images/human_data/faces/' + human_name + '/'


    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.transpose(1, 0, 2)


        size = (500, 500)
        images = facetrimming.face_check_from_ndarr(frame, size)

        for im in images:
            name = hashlib.md5(str(datetime.datetime.utcnow()).encode("utf-8")).hexdigest()
            cv2.imwrite(save_folder + name + '.png', im)

    cap.release()
