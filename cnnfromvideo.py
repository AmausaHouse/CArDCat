import os
import shutil

import const
import videoread
import getcontours
import preprocessing
import cnn

def main():

    names = []

    for name, movies in const.name_movies:

        names.append(name)

        path = './images/human_data/faces/' + name
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        for movie in movies:
            # videoread.videoread(name, movie)
            g = getcontours.GetContours()
            g.make_faces_picture(name, movie)

        path = './images/human_data/preprocessed_faces/' + name
        if os.path.exists(path):
            shutil.rmtree('./images/human_data/preprocessed_faces/' + name)
        os.mkdir('./images/human_data/preprocessed_faces/' + name)

        preprocessing.preprocessing(name)

    cnn.cnn(names)
    

if __name__ == '__main__':
    main()
