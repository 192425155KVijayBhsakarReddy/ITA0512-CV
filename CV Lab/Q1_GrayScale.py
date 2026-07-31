import cv2
import numpy as np
img=cv2.imread("images/input.jpg")
if img is None: raise FileNotFoundError("images/input.jpg not found")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
cv2.imshow("Original | Gray",np.hstack((img,gray)))
cv2.waitKey(0);cv2.destroyAllWindows()
