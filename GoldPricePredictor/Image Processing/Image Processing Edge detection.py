import cv2
import numpy as np
from matplotlib import pyplot as plt

#doorbin ro mikhanad
cap=cv2.VideoCapture(0)
while True:
    #tak tak frame ha ro migirim
    _,frame=cap.read()
    #laplasian baraye peyda kardan labeha
    laplacian=cv2.Laplacian(frame,cv2.CV_8U)

    #CV_8U= mahdoodeii ke bayad rrange ro bedahad, 1=mehvar X , 0=mehvar y, ksize=size pixle ha
    sobelx=cv2.Sobel(frame,cv2.CV_8U,1,0,ksize=5)
    #tamarkoz rooye mehvar y ha:
    sobely=cv2.Sobel(frame,cv2.CV_8U,0,1,ksize=5)
    #noe digar tashkhis labe
    canny=cv2.Canny(frame,100,200)


    cv2.imshow('Orginal',frame)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('canny',canny)
    #27=Esc
    if (cv2.waitKey(5) & 0Xff)==27:
        break

cv2.destroyAllWindows()
#bastan resource ha(doorbin):
cap.release()