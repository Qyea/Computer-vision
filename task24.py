import cv2
import numpy as np

OriginalImage = cv2.imread("abraham.jpg")
cv2.imshow("Original Image", OriginalImage)

makredDamages = cv2.imread("mask.jpg", 0)  # gray scale

# УБИРАЕМ ЛИНИИ ЗАЛОМА
# create a mask with threshhold
ret, thresh = cv2.threshold(makredDamages, 254, 255, cv2.THRESH_BINARY)

# make the lines thicker
kernel = np.ones((7, 7), np.uint8)
mask = cv2.dilate(thresh, kernel, iterations=1)

restoredImage = cv2.inpaint(OriginalImage, mask, 3, cv2.INPAINT_TELEA)
cv2.imshow("Image without Lines", restoredImage)

# ДОБАВЛЯЕМ КОНТРАСТ
# Увеличение контрастности
alpha = 1.2  # Коэффициент контрастности (1.0 - оригинальная яркость)
beta = -20     # Коэффициент яркости
# формула: output_pixel_value = input_pixel_value * alpha + beta
enhanced_image = cv2.convertScaleAbs(
    restoredImage, alpha=alpha, beta=beta)

# УБИРАЕМ ШУМ
# Билатеральный фильтр для удаления шума (можно также Гауссом, медианным, но этот лучше работает)
denoised_image = cv2.bilateralFilter(
    enhanced_image, d=3, sigmaColor=75, sigmaSpace=75)

cv2.imshow("output_image", denoised_image)
cv2.waitKey(0)

cv2.destroyAllWindows()