import cv2

# Загрузка изображения
image = cv2.imread('find.jpg')

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение алгоритма обнаружения границ Canny
edges = cv2.Canny(gray_image, 50, 150)

# Нахождение контуров объектов
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Выделение квадратов
squares = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) == 4:  # Проверка, является ли контур четырехугольником
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        squares.append(box)

# Вывод координат квадратов
for square in squares:
    for point in square:
        x, y = point
        print(f"Координаты точки: x = {x}, y = {y}")

# Рисование контуров квадратов на исходном изображении
cv2.drawContours(image, squares, -1, (0, 255, 0), 2)

# Вывод результатов
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()