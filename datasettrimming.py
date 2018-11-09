import datetime
import hashlib
import glob

import cv2

import facetrimming
import videoread

def datasettrimming(human_name):

    save_folder = './images/human_data/faces/' + human_name + '/'

    def get_learningdata(image_path):

        images = facetrimming.face_check(image_path)

        for i in range(len(images)):
            name = hashlib.md5(str(datetime.datetime.utcnow()).encode("utf-8")).hexdigest()
            cv2.imwrite(save_folder + name + '.png', images[i])

    image_pathes  = glob.glob('./images/human_data/photos/'+ human_name + '/*.jpg', recursive=True)
    image_pathes += glob.glob('./images/human_data/photos/'+ human_name + '/*.png', recursive=True)

    movie_pathes  = glob.glob('./images/human_data/photos/'+ human_name + '/*.mov', recursive=True)
    movie_pathes += glob.glob('./images/human_data/photos/'+ human_name + '/*.mp4', recursive=True)

    for image_path in image_pathes:
        get_learningdata(image_path)

    for movie_path in movie_pathes:
        videoread.videoread(human_name, movie_path)
