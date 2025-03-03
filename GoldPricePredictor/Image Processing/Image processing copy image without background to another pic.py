import cv2
import numpy as np
from matplotlib import pyplot as plt

#tarkib do aks
img=cv2.imread('C:\\PPpy\l.JPG',cv2.IMREAD_COLOR )
img2=cv2.imread('C:\\PPpy\ll.jpg',cv2.IMREAD_COLOR )

#danestan andaze satr va sotoon haye aks dovom
rows,cols,channels=img.shape

#entekhab bakhshi az aks aval ke aks dovom roosh miad [0:rowa]=az 0 ta rows aks aval
roi=img[0:rows,0:cols]

#aks ro grayscale mikonim
img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#sakht yek mask baraye hazve yek range rangi mesle pas zamine, THRESH_BINARY=agar balaye 220 bood 1 begir agar paeen 220 bood 0, inv= baraks mikonad
ret,mask=cv2.threshold(img2gray,120,155,cv2.THRESH_BINARY)

#baraks kardan mask
mask_inv=cv2.bitwise_not(mask)

#faghat bacground ro begire
img_bg=cv2.bitwise_and(roi,roi, mask=mask)

#forground ro joda mikone
img_fg=cv2.bitwise_and(img2,img2, mask=mask_inv)

#hardo aks ro rooye ham mindazim
dst=cv2.add(img_bg,img_fg)

#dar kodoomghesmat aks aval bedoon background ro bendaze rooye aks dovom
img[0:rows,0:cols]=dst

cv2.imshow('imgbg',img_bg)
cv2.imshow('imgfg',img_fg)
cv2.imshow('dst',dst)
cv2.imshow('imgnahaii',img)
cv2.imshow('frame',img2)
cv2.imshow('mask',mask)
cv2.imshow('mask_inv',mask_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()