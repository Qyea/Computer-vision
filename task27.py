import cv2


def detect_license_plate(input_image_path, output_image_path):
   # Загрузка изображения
   image = cv2.imread(input_image_path)
   cv2.imshow("image", image)
   

   # Преобразование изображения в оттенки серого
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


   # Использование каскадного классификатора для детекции лицевых частей (номерных знаков)
   plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
   plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
   i = 0
   # Отметка номерных знаков на изображении и вырезание областей в прямоугольниках
   for (x, y, w, h) in plates:
       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


       # Вырезание области изображения, соответствующей текущему прямоугольнику
       cropped_image = image[y:y + h, x:x + w]


       # Сохранение вырезанного изображения в файл
       cv2.imshow(f"cropped_image_{i}.jpg", cropped_image)
       i += 1


   # Сохранение результата
   cv2.imshow('output', image)
   cv2.waitKey(0)

   cv2.destroyAllWindows()




input_image_path = "27.jpg"
output_image_path = "output_with_license_plate.jpg"

detect_license_plate(input_image_path, output_image_path)
