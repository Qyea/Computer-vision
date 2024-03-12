import cv2

## Read
img = cv2.imread("sh3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## threshold and find contours
ret, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
cnts= cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

## Find the max-area-contour
cnt = max(cnts, key=cv2.contourArea)

## Approx the contour
arclen = cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, 0.002*arclen, True)

## Draw and output the result
for pt in approx:
   cv2.circle(img, (pt[0][0],pt[0][1]), 3, (0,255,0), -1, cv2.LINE_AA)

msg = "Total: {}".format(len(approx)//2)
cv2.putText(img, msg, (20,40),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)

## Display
cv2.imshow("res", img);cv2.waitKey()