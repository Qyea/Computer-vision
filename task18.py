import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lines(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Применение оператора Canny для обнаружения границ
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Создание маски
    mask = np.zeros_like(edges)
    height, width = image.shape[:2]
   
    # Обрезка интересующей области в которой будем искать линии (треугольник)
    pts = np.array([[0, height], [width // 2, height // 2], [width, height]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
   
    # Применение маски к границам
    masked_edges = cv2.bitwise_and(edges, mask)
   
    # Применение преобразования Хафа для обнаружения прямых линий
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
   
    # Фильтрация и отбрасывание лишних линий
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
       
        # Отбрасывание линий с углом наклона вне заданного диапазона
        if abs(angle) < 30 or abs(angle) > 60:
            continue
       
        filtered_lines.append(line)
   
    # Нанесение обнаруженных линий на изображение
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
   
    return image

# Загрузка изображения
image = cv2.imread('18.jpg')

# Вызов функции для обнаружения прямых линий и фильтрации
result = detect_lines(image)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)

# Отображение результата
plt.imshow(result)
plt.waitforbuttonpress(0)