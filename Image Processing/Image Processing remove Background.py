import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#hazf pas zamine

cap=cv2.VideoCapture(0)

#har frame  moghayese mikonad va taghirat ro moshakhas mikonad
fg=cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame=cap.read()
    #fg ra rooye hame frame ha emaal mikonad
    fmask=fg.apply(frame)
    cv2.imshow('orginal',frame)
    cv2.imshow('fg',fmask)

    k=cv2.waitKey(27) & 0Xff
    if(k==27):
        break
cv2.destroyAllWindows()
cap.release()        
