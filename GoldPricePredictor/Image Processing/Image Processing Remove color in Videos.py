import cv2
import numpy as np
from matplotlib import pyplot as plt

#doorbin ro mikhanad
cap=cv2.VideoCapture(0)
while True:
    #tak tak frame ha ro migirim
    _,frame=cap.read()
    #HSL va HSV = H=noe rang , S=shedat roshanaii , S=shedat khode rang(rangi siahsefid)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #filter kardane range rang:
    lower_red=np.array([0,50,50])
    upper_red=np.array([10,255,255])

    #tarif mask ke az rooye frame ha che rang rangi ro joda konad
    mask=cv2.inRange(hsv,lower_red,upper_red)

    #baraks kardan mask:
    mask=cv2.bitwise_not(mask)
    res=cv2.bitwise_and(frame,frame,mask=mask)

    #namayesh frame:
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #27=Esc
    if (cv2.waitKey(5) & 0Xff)==27:
        break

cv2.destroyAllWindows()
#bastan resource ha(doorbin):
cap.release()