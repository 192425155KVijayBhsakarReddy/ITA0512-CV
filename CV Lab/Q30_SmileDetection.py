import cv2,numpy as np
img=cv2.imread("images/image1.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
smile=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")
out=img.copy()
for(x,y,w,h) in smile.detectMultiScale(gray,1.8,20):
    cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow("Original | Smile Detection",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
