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
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #alave bar namayesh zakhiro konad
    out.write(frame)
    # esme panjere
    cv2.imshow('frame', gray)
    #chegoone kharej shavim: (0XFF= meghdar click)
    if cv2.waitKey(1) & 0Xff==ord('q'):
        break

#release video:
cap.release()
#file ro zikhire konad bad bebandad
out.release()
# cv2.destroyAllWindow()

#---------------------------------------------------------
#video rangi va rang RGB
import cv2
import numpy as np
from matplotlib import pyplot as plt
#video
#estefadeh az web cam default shomare 1
cap=cv2.VideoCapture(0)
# chon video aks haye poshte ham ast az while estefadeh mikonim
while(True):
    res,frame=cap.read()
    #video grayscale tartibesh BGR ast na RGB
    color=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    # esme panjere va rangesh
    cv2.imshow('frame', color)
    #chegoone kharej shavim: (0XFF= meghdar click)
    if cv2.waitKey(1) & 0Xff==ord('q'):
        break
#release video:
cap.release()
# cv2.destroyAllWindow()
#------------------------------------------------------