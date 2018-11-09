import os
import datetime
import hashlib
import glob

import matplotlib.pyplot as plt
import cv2

def datasetselect(name):

    open_folder = './images/human_data/faces/' + name + '/'

    image_pathes = glob.glob(open_folder + "*.png")

    for image_path in image_pathes:
        i = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        plt.pause(.01)

        print('2: remove => {}'.format(image_path))
        if input() == "2":
            os.remove(image_path)
            print('removed')

    plt.close("all")
