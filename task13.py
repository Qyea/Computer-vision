import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
# С комментами, чтобы было понятнее, но таска вроде простая

image = cv2.imread('image.jpg', 0)

# Тут получаем гистограмму яркости
histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=[0, 256])

# Вычислили значение порога методом Отцу
threshold = threshold_otsu(image)

plt.figure()
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram)
plt.axvline(x=threshold, color='r', linestyle='--', linewidth=1, label='Threshold (Otsu)') # Отметили порог
plt.legend()

# Применение порога к изображению
binary_image = np.where(image > threshold, 255, 0)

plt.figure()
plt.title('Image with Otsu Threshold')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()