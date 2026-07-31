import cv2, numpy as np
img=cv2.imread("images/input.jpg")
if img is None: raise FileNotFoundError("images/input.jpg not found")
blur=cv2.GaussianBlur(img,(15,15),0)
cv2.imshow("Original | Blur",np.hstack((img,blur)))
cv2.waitKey(0);cv2.destroyAllWindows()
