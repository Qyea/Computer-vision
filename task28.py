import cv2
import numpy as np

image_file = "28.jpg"
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

output = img.copy()
for idx, contour in enumerate(contours):
   if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 1000:
       (x, y, w, h) = cv2.boundingRect(contour)
       if hierarchy[0][idx][3] == 0:
           cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)


cv2.imshow("Input", img)
cv2.drawContours(gray, contours, -1, (255, 255, 255), 1)
cv2.imshow("Enlarged", img_erode)
cv2.imshow("Output", output)
cv2.waitKey(0)