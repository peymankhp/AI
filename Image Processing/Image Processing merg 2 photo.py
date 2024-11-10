import cv2
import numpy as np
from matplotlib import pyplot as plt

#ravesh aval andakhtan 2 aks rooye ham ba size yeksaan
img=cv2.imread('C:\\PPpy\l.jpg',cv2.IMREAD_COLOR )
img2=cv2.imread('C:\\PPpy\ll.jpg',cv2.IMREAD_COLOR )
add=img+img2
cv2.imshow('add',add)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ravesh dovom andakhtan 2 aks rooye ham
add2=cv2.add(img,img2)
cv2.imshow('add',add2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ravesh dovom andakhtan 2 aks rooye ham ba tanzime mizan tasir har aks
add3=cv2.addWeighted(img,0.4,img2,0.7,0)#0=gamma
cv2.imshow('add',add3)
cv2.waitKey(0)
cv2.destroyAllWindows()