import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('input_image.jpg', 0)

threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Применение морфологического замыкания для объединения разорванных фрагментов
kernel = np.ones((5, 5), np.uint8)
closed_image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

# Вывод результатов
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', threshold)
cv2.imshow('Closed Image', closed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()