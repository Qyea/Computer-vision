import cv2
import numpy as np

book_image = cv2.imread('book_image.jpg')
template_image = cv2.imread('template_image.jpg')

book_gray = cv2.cvtColor(book_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# Нахождение ключевых точек и дескрипторов на обоих изображениях
orb = cv2.ORB_create()
book_keypoints, book_descriptors = orb.detectAndCompute(book_gray, None)
template_keypoints, template_descriptors = orb.detectAndCompute(template_gray, None)

# Создание объекта матчера
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Поиск соответствий между дескрипторами
matches = matcher.match(book_descriptors, template_descriptors)

# Сортировка соответствий по расстоянию
matches = sorted(matches, key=lambda x: x.distance)

num_good_matches = 50
good_matches = matches[:num_good_matches]

# Извлечение координат ключевых точек для хороших соответствий
book_points = np.float32([book_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
template_points = np.float32([template_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Выполнение проективного преобразования
M, _ = cv2.findHomography(book_points, template_points, cv2.RANSAC)

# Результат обработки 
aligned_image = cv2.warpPerspective(book_image, M, (template_image.shape[1], template_image.shape[0]))


# Для нормального вывода уменьшаем изображение, а то не помещается...
output_width = 500 
output_height = int(output_width * (aligned_image.shape[0] / aligned_image.shape[1]))
resized_image = cv2.resize(aligned_image, (output_width, output_height))

cv2.imshow('Aligned Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()