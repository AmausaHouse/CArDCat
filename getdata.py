import glob

import numpy as np

import cv2

import const

dataset_folder = './images/learning_data/png/'
target_folder = './images/human_data/'

def get_image_pathes(target_name, get_preprocessed = False):

    not_match_image_pathes = glob.glob(dataset_folder + "*.png")
    target_pathes = glob.glob(target_folder.replace('faces', 'preprocessed_faces' if get_preprocessed else 'faces') + "*/*.png", recursive=True)

    not_match_image_pathes += list(filter(lambda x:x.find(target_name) == -1, target_pathes))
    match_image_pathes = list(filter(lambda x:x.find(target_name) != -1, target_pathes))

    return match_image_pathes, not_match_image_pathes

def make_dataset(ids):

    # でーたせっと
    pathes = glob.glob(dataset_folder + "*.png")
    np.random.shuffle(pathes)
    pathes = pathes[:min(len(pathes), const.testdata_max)]

    # 本質の方
    # pathes += glob.glob(target_folder + "*/png/*/*.png", recursive=True)
    for nam in ids:
        pathes += glob.glob(target_folder + nam + "/png/*/*.png", recursive=True)[:3000]

    vfunc = np.vectorize(lambda x: x / 255.0)

    size = len(pathes)
    label_size = len(ids)

    np.random.shuffle(pathes)

    data = np.zeros((size, 64, 64, 3), dtype='float32')
    label = np.zeros((size, label_size + 1), dtype='float32')

    count = [0 for i in range(label_size + 1)]

    for i in range(size):
        flag = label_size

        data[i] = vfunc(cv2.imread(pathes[i]).astype('float32'))

        # ファイル名からラベルを張る
        for j in range(label_size):
            if pathes[i].find(ids[j]) != -1:
                flag = j
                break

        count[flag] += 1
        label[i][flag] = 1

    print('label count : {}'.format(count))

    return label_size, data, label
