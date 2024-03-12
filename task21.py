from PIL import Image
import numpy as np
from PIL import ImageOps
import cv2

# Укажите путь к изображению
image_path = "21.jpg"

# Загрузка изображения
image = Image.open(image_path)

# Преобразование в черно-белый формат с пороговым значением
_, binary_image = cv2.threshold(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), 128, 255, cv2.THRESH_BINARY)

# Вам может понадобиться использовать методы обработки изображений
# или контурного анализа для оценки сложности формы объекта. Например,
# можно использовать функцию нахождения контуров:
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Оценка сложности формы
complexity_score = len(contours)
print(complexity_score)

cv2.imshow('output', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()