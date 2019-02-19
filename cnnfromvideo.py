import const
import getcontours
import cnn

def main():

    ids = []

    g = getcontours.GetContours()

    for id, movie_ids in const.name_movies:

        ids.append(id)

        for movie in movie_ids:
            g.make_faces_picture(id, movie)

    cnn.cnn(ids)


if __name__ == '__main__':
    main()
