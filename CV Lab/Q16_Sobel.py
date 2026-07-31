import cv2,numpy as np
img=cv2.imread("images/input.jpg")
g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sx=cv2.Sobel(g,cv2.CV_64F,1,0)
sx=cv2.cvtColor(cv2.convertScaleAbs(sx),cv2.COLOR_GRAY2BGR)
cv2.imshow("Original | Sobel",np.hstack((img,sx)))
cv2.waitKey(0);cv2.destroyAllWindows()
