import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('squares.jpg')


# Преобразование изображения в цветовое пространство HSV - говорят, что так лучше
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Определение диапазона значений HSV для оранжевого цвета
lower_orange = np.array([10, 50, 50])
upper_orange = np.array([18, 255, 255])


# Создание маски оранжевого цвета
mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Применение операции поиска контуров
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Поиск и отображение прямоугольников
for contour in contours:
   x, y, w, h = cv2.boundingRect(contour)
   cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


output_width = 500 
output_height = int(output_width * (image.shape[0] / image.shape[1]))
resized_image = cv2.resize(image, (output_width, output_height))

# Вывод результатов
cv2.imshow('Original Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()