
#khandane aks asli
img=cv2.imread('C:\\PPpy\st.jpg')
img2=cv2.imread('C:\\PPpy\st2.jpg')

#detctor olgoo
orb=cv2.ORB_create()

# 3tarif key point
kp1,des1=orb.detectAndCompute(img2,None)
kp2,des2=orb.detectAndCompute(img,None)

#mach kardan key point ha
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

#che noghate az detect ha match hastand
matches=bf.match(des1,des2)

#sort kardan noghat match shode
matches=sorted(matches, key=lambda x: x.distance)

img_out=cv2.drawMatches(img,kp1,img2,kp2,matches[:10],None, flags=2)

cv2.imshow('corners', img_out)
cv2.waitKey(0)