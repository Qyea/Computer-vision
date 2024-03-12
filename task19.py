import cv2

cap = cv2.VideoCapture(0)

while True:
   ret, img = cap.read()

   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
   blur = cv2.GaussianBlur(gray, (7, 7), 0)
   
   canny = cv2.Canny(blur, 0, 120)
   ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
   cv2.imshow('Video feed', mask)
   # press Enter to exit
   if cv2.waitKey(1) == 13:
       break


cap.release()
cv2.destroyAllWindows()
