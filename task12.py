import cv2
import numpy as np


image = cv2.imread('boat-rotated.jpg')


# преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# обнаружение углов алгоритмом Харриса
# (параметры можно менять в зависимости от изображения -
# maxCorners - максимальное количество угловых точек для обнаружения,
# qualityLevel - минимальное качество углов, 
# minDistance - минимальное расстояние между углами,
# blockSize - размер блока для обнаружения углов)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=20, blockSize=11)


if corners is not None:
   # преобразование данных в целочисленные значения
   corners = np.intp(corners)  


   # определение осей путем сингулярного разложения (SVD)
   _, _, vt = np.linalg.svd(corners[:, 0, :] - np.mean(corners[:, 0, :], axis=0)) 
   # вычисление угла поворота
   angle = np.rad2deg(np.arctan2(vt[0, 1], vt[0, 0]))   
else:
   angle = 0


(h, w) = image.shape[:2]
center = (w // 2, h // 2)   # определение центра изображения
# создание матрицы поворота
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) 
aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) 
# применение матрицы поворота

output_width = 500 
output_height = int(output_width * (aligned_image.shape[0] / aligned_image.shape[1]))
resized_image = cv2.resize(aligned_image, (output_width, output_height))
image  = cv2.resize(image, (output_width, output_height))

cv2.imshow('Original', image)
cv2.imshow('Aligned', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
