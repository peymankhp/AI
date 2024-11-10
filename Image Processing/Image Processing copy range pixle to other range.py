import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('C:\\PPpy\love.JPG',cv2.IMREAD_COLOR )
#namayesh etelaat rang(BGR) yek pixel khas:
px=img[200,200]
print(px)
#khoondan yek range az pixelha
pxrange=img[100:110,100:110]
print(pxrange)
#taghirat rooye tasvir [255,255,255]=white color
img[100:310,100:310]=[255,255,255]
cv2.imshow('C:\\PPpy\love.JPG',img)
cv2.waitKey(0)
# jabejaii va copy yek range pixel dar range digar 
changepixel=img[300:410,310:380]
img[500:610,510:580]=changepixel
cv2.imshow('C:\\PPpy\love.JPG',img)
cv2.waitKey(0)