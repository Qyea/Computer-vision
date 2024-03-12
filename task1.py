import cv2

image_path = "star.jpg"
original_image = cv2.imread(image_path)

cv2.imshow("Without contours", original_image)

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Применение бинаризации для выделения объекта
_, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Поиск контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(original_image, contours, -1, (0, 0, 255), 2)

size = 0

# Перебор найденных контуров
for contour in contours:
   # Вычисление площади контура и добавление к общей площади
   area = cv2.contourArea(contour)
   size += area
   x, y, w, h = cv2.boundingRect(contour)
   cv2.putText(original_image, str(area), (x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Вывод результата
print("Площадь объекта: {} пикселей".format(size))

cv2.imshow("With contours", original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()