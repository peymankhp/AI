import cv2
import numpy as np
from matplotlib import pyplot as plt
#khoondan aks va halat haye khondan aks siah sefid ya rangi
img=cv2.imread('C:\\PPpy\love.JPG', cv2.IMREAD_GRAYSCALE)
#namayesh aks
cv2.imshow('C:\\PPpy\love.JPG',img)
#ba click biad biroon
cv2.waitKey(0)
#aks too hafeze namoone va azad beshe resource ha
# cv2.destroyWindow('love.JPG')
plt.imshow(img,cmap='gray',interpolation='bicubic')
#rasm khat bar rooye tasvir
plt.plot([100,200],[200,300],'r',linewidth=5)
plt.show()
cv2.imwrite('C:\\PPpy\loveGray.JPG',img)