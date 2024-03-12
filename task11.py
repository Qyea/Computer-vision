import cv2

# Загрузка изображения
image = cv2.imread('img.jpg')
cv2.imshow('Original', image)

# Преобразование в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Пороговое значение яркости
threshold_value = 95

# Применение порогового значения
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# Применение операций морфологического преобразования
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Нахождение контуров пораженных мест
contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Отрисовка контуров на исходном изображении
cv2.drawContours(image, contours, -1, (0, 0, 0), 1)

# Отображение результатов
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()