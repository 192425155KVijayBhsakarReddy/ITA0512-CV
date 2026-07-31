import cv2,numpy as np
img=cv2.imread("images/face.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
eye=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
out=img.copy()
for(x,y,w,h) in eye.detectMultiScale(gray,1.3,5):
    cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("Original | Eye Detection",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
