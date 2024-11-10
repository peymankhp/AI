import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("C:\\PPpy\sample.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ravesh aval threshold
#255= tahe range threshold
ret,threshold=cv2.threshold(img,16,255,cv2.THRESH_BINARY)

#ravesh dovom adaptiveThreshold, 115=block size , 1=destination
th=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)


cv2.imshow('page',img)
cv2.imshow('paget',threshold)
cv2.imshow('pageth',th)
cv2.waitKey(0)
cv2.destroyAllWindows()