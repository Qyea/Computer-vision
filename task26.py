import cv2
import numpy as np
import matplotlib.pyplot as plt

coin = cv2.imread("coins.jpg")
"""cv2.imshow('Aligned Image', coin)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
plt.figure(), plt.title("Original"), plt.imshow(coin), plt.axis("off");

# Размытие
coin_blur = cv2.medianBlur(src=coin, ksize=15)
plt.figure(), plt.title("Low Pass Filtering (Blurring)"), plt.imshow(coin_blur), plt.axis("off");

# Изображение в оттенках серого
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.title("Gray Scale"), plt.imshow(coin_gray, cmap="gray"), plt.axis("off");

# Бинаризация
ret, coin_thres = cv2.threshold(src=coin_gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)
plt.figure(), plt.title("Binary Threshold"), plt.imshow(coin_thres, cmap="gray"), plt.axis("off");

kernel = np.ones((5,5), np.uint8)
# Морфологическое размыкание
opening = cv2.morphologyEx(coin_thres, cv2.MORPH_OPEN, kernel=kernel, iterations=2)

plt.figure(), plt.title("Opening"), plt.imshow(opening, cmap="gray"), plt.axis("off");
# Вычисляем расстояние между объектами и строим скелет изображения
dist_transform = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)

plt.figure(), plt.title("Distance Transform"), plt.imshow(dist_transform, cmap="gray"), plt.axis("off");
# Получаем область переднего плана
ret, sure_foreground = cv2.threshold(src=dist_transform, thresh=0.4*np.max(dist_transform), maxval=255, type=0)

plt.figure(), plt.title("Fore Ground"), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off");
# Получаем область заднего плана
sure_background = cv2.dilate(src=opening, kernel=kernel, iterations=1) #int

sure_foreground = np.uint8(sure_foreground) # change its format to int
# Получаем разность фона и изображений на переднем плане
unknown = cv2.subtract(sure_background, sure_foreground)

plt.figure(), plt.title("BackGround - ForeGround = "), plt.imshow(unknown, cmap="gray"), plt.axis("off");
# Присваиваем метки связным компонентам
ret, marker = cv2.connectedComponents(sure_foreground)

marker = marker + 1

marker[unknown == 255] = 0 # White area is turned into Black to find island for watershed

plt.figure(), plt.title("Connection"), plt.imshow(marker, cmap="gray"), plt.axis("off");
# Применяем watershed
marker = cv2.watershed(image=coin, markers=marker)

plt.figure(), plt.title("Watershed"), plt.imshow(marker, cmap="gray"), plt.axis("off");

# Находим контуры
contour, hierarchy = cv2.findContours(image=marker.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

# Итерируемся по контурам и рисуем их на исходном изображении
for i in range(len(contour)):

    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image=coin,contours=contour,contourIdx=i, color=(255,0,0), thickness=2)

plt.figure(figsize=(7,7)), plt.title("After Contour"), plt.imshow(coin, cmap="gray"), plt.axis("off");
plt.show()