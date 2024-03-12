import cv2 
import numpy as np
#применение эрозии для "разрыва" изображение на франменты
def apply_erosion(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img = cv2.erode(img,kernel,iterations = 1)
    return img

#применение дилатации для соединения разорванных фрагментов
def apply_dilatation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img = cv2.dilate(img,kernel,iterations = 2)
    return img

img = cv2.imread('23.jpg', cv2.IMREAD_GRAYSCALE)
#бинаризация изображения с использованием otsu
ret,binarized_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

#эрозия и дилатация правильно работают если фон черный, вот и делаем пиксели которые были черными -  белыми, и наоборот
binarized_img = 255 - binarized_img

eroded = apply_erosion(binarized_img)

merged = apply_dilatation(eroded)

cv2.imshow('source', binarized_img)
cv2.imshow('broken', eroded)
cv2.imshow('merged', merged)

cv2.waitKey(0)
cv2.destroyAllWindows()
