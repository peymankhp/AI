import cv2
import numpy as np
from matplotlib import pyplot as plt

#video
#estefadeh az web cam default shomare 1
cap=cv2.VideoCapture(0)

#zakhirevideo: XVID  =noe codec
fourcc=cv2.VideoWriter_fourcc(*'XVID')

#makan zakhire va sorat frame va hajme zakhire
out=cv2.VideoWriter('video.avi', fourcc, 24.0, (680, 480))

# chon video aks haye poshte ham ast az while estefadeh mikonim
while(True):
    res,frame=cap.read()
    #video grayscale tartibesh BGR ast na RGB
    # frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    #rasm khat 255,0,0=Blue
    cv2.line(frame,(100,200),(200,400),(255,0,0),5)
    #rectangle:
    cv2.rectangle(frame,(100,200),(200,400),(255,255,0),10)
    #circle:
    cv2.circle(frame,(300,400),70,(120,0,125),7)
    #polyline
    # noghat
    pts=np.array([[10,12],[22,30],[45,12],[46,20],[50,20],[33,77]])
    #True or False yani khat ro bebandad 5=zekhamat
    cv2.polylines(frame,[pts],True,(60,40,20),5)
    #neveshtan rooye video, Font:
    font=cv2.FONT_HERSHEY_PLAIN
    #2 = andaze font va 1=zekhamat
    cv2.putText(frame,'Hey this is TEST',(70,70),font,2,(100,50,50),1)
    #alave bar namayesh zakhiro konad
    out.write(frame)
    # esme panjere
    cv2.imshow('frame', frame)
    #chegoone kharej shavim: (0XFF= meghdar click)
    if cv2.waitKey(1) & 0Xff==ord('q'):
        break

#release video:
cap.release()
#file ro zikhire konad bad bebandad
out.release()
# cv2.destroyAllWindow()