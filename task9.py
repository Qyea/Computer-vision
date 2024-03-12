import cv2
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

COUNT = 0

def detection(image):
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # Бинаризируем изображение (порог - 100)
   _, binary = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

   # Операция раскрытия (последовательное применение эрозии и дилатации) для очистки изображения от шумов
   kernel = np.ones((5, 5), dtype=np.uint8)
   opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

   # Карта расстояний для нахождения зоны, в которой точно есть зерно (евклидово расстояние)
   dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
   _, sure_foreground = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, cv2.THRESH_BINARY)
   sure_foreground = np.uint8(sure_foreground)

   # Промежуток, в котором мы не уверены, есть ли зерно или нет (вычитание)
   unknown = cv2.subtract(opening, sure_foreground)

   # Создаем матрицу с размером исходного изображения, где каждому "пикселю"
   # принадлежит номер компоненты, которой он принадлежит
   # Изначально фону маркеруем нулевой номер, но в водоразделе тогда эта зона будет считаться неизвестной.
   # Поэтому нумеруем все компоненты с +1 и зону unknown маркируем нулевой
   _, markers = cv2.connectedComponents(sure_foreground)
   markers = markers + 1
   markers[unknown == 255] = 0
   # Применяем водораздел (разделит зерна)
   markers = cv2.watershed(image, markers)

   # Генерим массив с цветом для каждой компоненты
   colors = np.random.randint(0, 255, size=(np.max(markers) + 1, 3), dtype=np.uint8)
   # И теперь в маркированном массиве каждой компоненте присвоится нужный цвет
   colored_cells = colors[markers]

   # Тут количество зерен считается, если нужно
   global COUNT
   COUNT = len(np.unique(markers)) - 1

   return colored_cells


# Чтение изображения
original_image = None
file_path = 'sample10.jpg'
if file_path:
   original_image = cv2.imread(file_path)

# Выделяем зерна
processed_image = detection(original_image)

# Выводим на графике
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].set_title("Original")
axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].axis('off')
axes[1].set_title("Processed")
axes[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
axes[1].axis('off')
plt.show()
