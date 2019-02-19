import numpy as np
import os
import glob
import subprocess
import hashlib
import datetime

import cv2

import getdata

import const


def preprocessing(save_folder, picture_id, data):

    def make_contrast(min_t, max_t, flag):
        diff_t = max_t - min_t
        v = np.zeros((256, 1), dtype = 'uint8')

        for i in range(256):
            v[i][0] = (0 if i < min_t else 255 if i >= max_t else 255 * (i - min_t) / diff_t) if flag else (min_t + i * diff_t / 255)

        return v

    vectorize_noise = np.vectorize(lambda x:max(0, min(255, int(x + np.random.normal(0, 25)))))

    functions = [
            lambda x: cv2.flip(x, 1),
            lambda x: cv2.GaussianBlur(x, (7, 7), 0),
            lambda x: cv2.LUT(x, make_contrast(np.random.rand() * 95, 160 + np.random.rand() * 95, np.random.normal(0, 1) > 0.5)),
            lambda x: vectorize_noise(x)
            ]

    def scratch_image(origin_image):

        scratch_images = []

        if not const.use_preprocessing:
            return [origin_image]

        for i in range(1 << len(functions)):

            scratch_image = origin_image

            for j in range(len(functions)):
                if i & (1 << j):
                    scratch_image = functions[j](scratch_image)

            scratch_images.append(scratch_image)

        return scratch_images

    images = scratch_image(data)

    for index, img in enumerate(images):
        path = save_folder + '/' + picture_id + '_' + str(index) + '.png'
        cv2.imwrite(path, img)
