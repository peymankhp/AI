import cv2
import numpy as np
from matplotlib import pyplot as plt

#khandane aks asli
img=cv2.imread('C:\\PPpy\T.jpg')
#tabdil be siaho sefid
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

imgtemp=cv2.imread('C:\\PPpy\TT.jpg',0)
#tool va arze ro tashkhis bedahim
w,h=imgtemp.shape[::-1]

#tashkhis patern dar aks, cv2.TM_CCOEFF_NORMED= ravesh tashkhis
res=cv2.matchTemplate(img_gray,imgtemp,cv2.TM_CCOEFF_NORMED)
#threshold=aastane shebahat
threshold= 0.8

#noghat peyda shode:
loc=np.where(res >= threshold)

#namayesh dadan shebahat ha
for pt in zip(*loc[::-1]):
    #mostatil bekeshe doresh, pt=noghat shoro va payan, (pt[0]+w)=noghte peyda shode be alave arzesh , 
    # (pt[1]+h)= noghte dovom be alave ertefaesh, (0,0,255) ghermez , 1= arze mostatil
    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),1)

cv2.imshow('peyman', img)
cv2.waitKey(0)
