import cv2
import numpy as np

img=cv2.imread('C:\\PPpy\4.jpg')
#255= tahe range threshold
ret,threshold=cv2.threshold(img,10,255,cv2.THRESH_BINARY)

cv2.imshow('page',img)
cv2.imshow('paget',threshold)
cv2.waitkey(0)
cv2.destroyAllWindows()















