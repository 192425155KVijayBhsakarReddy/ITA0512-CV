import cv2,numpy as np
img=cv2.imread("images/input.jpg")
gray=np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
dst=cv2.cornerHarris(gray,2,3,0.04)
out=img.copy(); out[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow("Original | Harris",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
