import cv2,numpy as np
img=cv2.imread("images/face.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
out=img.copy()
faces=face.detectMultiScale(gray,1.3,5)
for(x,y,w,h) in faces:
    cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
print("Number of Faces:",len(faces))
cv2.imshow("Original | Face Count",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
