import cv2
import numpy as np
from matplotlib import pyplot as plt


def contrast1(value):
   clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(8, 8))
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB
   l, a, b = cv2.split(lab)  # split on 3 channels
   l2 = clahe.apply(l)  # apply CLAHE to the L-channel
   lab = cv2.merge((l2, a, b))  # merge channels
   img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
   img = np.hstack((image, img))
   cv2.imshow('image', img)


def contrast2(alpha):
   img = cv2.convertScaleAbs(image, alpha=alpha / 10, beta=0)
   img = np.hstack((image, img))
   cv2.imshow("image", img)



def brightness(value):
   img = cv2.convertScaleAbs(image, alpha=1, beta=value)
   img = np.hstack((image, img))
   cv2.imshow('image', img)


image = cv2.imread('face.jpg')

cv2.namedWindow("image")
cv2.imshow("image", image)


cv2.createTrackbar("Contrast (alpha)", "image", 10, 30, contrast2)  # 1.0-3.0
cv2.createTrackbar('Brightness (beta)', 'image', 0, 100, brightness)


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
