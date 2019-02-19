import os
import shutil

import const
import getcontours
import preprocessing
import cnn

def main():

    names = []

    g = getcontours.GetContours()

    for id, movie_ids in const.name_movies:

        names.append(id)

        for movie in movie_ids:
            g.make_faces_picture(id, movie)

    cnn.cnn(names)


if __name__ == '__main__':
    main()
