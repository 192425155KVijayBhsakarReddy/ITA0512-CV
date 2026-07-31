import cv2,numpy as np
img=cv2.imread("images/input.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,th=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
th=cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)
cv2.imshow("Original | Segmentation",np.hstack((img,th)))
cv2.waitKey(0);cv2.destroyAllWindows()
