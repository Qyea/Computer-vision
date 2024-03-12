import cv2
import numpy as np

def convex(image):
    convex_image = image.copy()

    # Преобразуем изображение в оттенки серого
    img_gray = cv2.cvtColor(convex_image, cv2.COLOR_BGR2GRAY)

    # Бинаризируем изображение
    ret, binary_image = cv2.threshold(img_gray,50,255,0)

    # Ищем контуры
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # Для каждого контура находим выпуклую оболочку и рисуем ее
    # на исходном изображении.
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(convex_image, [hull], -1, (255, 0, 0), 2)

    return convex_image


image_path = 'cars.jpg'

image = cv2.imread(image_path)

if image is not None:
    convex_image = convex(image)
    cv2.imshow('Source image', image)
    cv2.imshow('Convex hull', convex_image)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to load image from path: {image_path}")
