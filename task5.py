import cv2
import numpy as np
from matplotlib import pyplot as plt


# Загрузка спутникового снимка
image_path = '5.jpg'
satellite_image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2GRAY)

# Применение оператора Лапласиана Гаусса (LoG)
log_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
log_image = cv2.Laplacian(log_image, cv2.CV_64F)

# Бинаризация изображения на основе порога
threshold_value = 23
_, binary_image = cv2.threshold(np.abs(log_image), threshold_value, 255, cv2.THRESH_BINARY)

# Используем маску для выделения только дорог
road_mask = cv2.inRange(gray_image, 155, 255)  
binary_image = cv2.bitwise_and(binary_image, binary_image, mask=road_mask)

# Скелетизация изображения
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
skel_image = np.zeros_like(binary_image)

while True:
    erode = cv2.erode(binary_image, element)
    temp = cv2.dilate(erode, element)
    temp = cv2.subtract(binary_image, temp)
    skel_image = cv2.bitwise_or(skel_image, temp)
    binary_image = erode.copy()


    if cv2.countNonZero(binary_image) == 0:
        break


plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('Original Image')
plt.subplot(133), plt.imshow(skel_image, cmap='gray'), plt.title('Skeleton Image')


plt.show()
