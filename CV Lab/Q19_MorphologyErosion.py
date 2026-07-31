import cv2,numpy as np
img=cv2.imread("images/input.jpg")
k=np.ones((5,5),np.uint8)
out=cv2.erode(img,k,1)
cv2.imshow("Original | Morph Erosion",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
