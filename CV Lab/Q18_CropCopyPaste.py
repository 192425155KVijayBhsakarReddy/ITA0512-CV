import cv2,numpy as np
img=cv2.imread("images/input.jpg")
out=img.copy()
roi=img[50:150,50:150]
out[200:300,200:300]=cv2.resize(roi,(100,100))
cv2.imshow("Original | ROI",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
