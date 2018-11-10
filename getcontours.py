import cv2

import matplotlib.pyplot as plt

import facetrimming
import const

color0 = (60, 60, 180)
color1 = (180, 180, 180)



image = cv2.imread(const.image_path)

def getcontours(image, reducation_per=1.7):
    rect = facetrimming.get_rect_from_ndarr(image)

    for r in rect:

        x_border = min(0, r[0] - int(r[2] * ((reducation_per - 1) / 2)))
        y_border = min(0, r[1] - int(r[2] * ((reducation_per - 1) / 2)))

        r[0] -= int(r[2] * ((reducation_per - 1) / 2)) + x_border
        r[1] -= int(r[3] * ((reducation_per - 1) / 2)) + y_border

        r[2] = min(image.shape[1] - r[0], int(r[2] * reducation_per) - x_border)
        r[3] = min(image.shape[0] - r[1], int(r[3] * reducation_per) - y_border)


        cv2.rectangle(image, tuple(r[0:2]),tuple(r[0:2]+r[2:4]), color0, thickness=25)
        cv2.rectangle(image, tuple(r[0:2]),tuple(r[0:2]+r[2:4]), color1, thickness=8)

        
        print(r)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.pause(.01)

    input()


getcontours(image)
