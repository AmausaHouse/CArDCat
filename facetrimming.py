import cv2
import const

def face_check(image_path, minsize=(25, 25)):

    image = cv2.imread(image_path)

    return face_check_from_ndarr(image, minsize)

def face_check_from_ndarr(image, minsize=(25, 25)):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(const.cascade_path)

    rect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=minsize)

    return rect_to_subarray(rect, image)


def get_rect_from_ndarr(image, minsize=(25, 25)):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(const.cascade_path)

    rect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=minsize)

    return rect


def rect_to_subarray(rect, image):
    resized_images = []

    for r in rect:
        resized_images.append(image[r[1]:r[1]+r[3],r[0]:r[0]+r[2]])

    return resized_images
