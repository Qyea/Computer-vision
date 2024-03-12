import cv2

# Загрузка изображения
image = cv2.imread('star.jpg')

cv2.imshow("Without contours", image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение бинаризации для выделения объекта
_, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Поиск контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

# Вычисление периметра объекта
perimeter = cv2.arcLength(contours[0], True)

cv2.imshow('binary', image)

print("Площадь объекта: {} пикселей".format(perimeter))
cv2.waitKey(0)