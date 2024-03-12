from deconvolution import Deconvolution
from PIL import Image
import cv2
import numpy as np

image = Image.open("image003.jpg")
new_width = 500
new_height = 500
img = image.resize((new_width, new_height))

decimg = Deconvolution(image=img, basis=[[174/255,31/255,43/255], [87/255, 89/255, 146/255]])
layer1, layer2= decimg.out_images(mode=[1, 2])

layer1 = np.array(layer1)
layer2 = np.array(layer2)

layer1 = cv2.cvtColor(layer1, cv2.COLOR_BGR2RGB)
layer2 = cv2.cvtColor(layer2, cv2.COLOR_BGR2RGB)

cv2.imshow('stump', layer1)
cv2.imshow('sights', layer2)
cv2.waitKey(0)
cv2.destroyAllWindows()