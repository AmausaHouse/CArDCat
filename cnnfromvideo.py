import os
import shutil

import const
import getcontours
import preprocessing
import cnn

def main():

    names = []

    for name, movies in const.name_movies:

        names.append(name)

        g = getcontours.GetContours()

        # facesを空にする
        path = './images/human_data/faces/' + name
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        # 動画からmake_faces_pictureをする
        for movie in movies:
            g.make_faces_picture(name, movie)

        path = './images/human_data/preprocessed_faces/' + name
        if os.path.exists(path):
            shutil.rmtree('./images/human_data/preprocessed_faces/' + name)
        os.mkdir('./images/human_data/preprocessed_faces/' + name)

        # preprocessingをする
        preprocessing.preprocessing(name)

    cnn.cnn(names)
    

if __name__ == '__main__':
    main()
