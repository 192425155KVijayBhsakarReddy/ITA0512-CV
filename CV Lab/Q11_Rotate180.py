import cv2,numpy as np
img=cv2.imread("images/input.jpg")
if img is None: raise FileNotFoundError("images/input.jpg not found")
rot=cv2.rotate(img,cv2.ROTATE_180)
cv2.imshow("Original | Rotate 180",np.hstack((img,rot)))
cv2.waitKey(0);cv2.destroyAllWindows()
