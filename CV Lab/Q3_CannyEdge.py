import cv2,numpy as np
img=cv2.imread("images/input.jpg")
if img is None: raise FileNotFoundError("images/input.jpg not found")
edge=cv2.Canny(img,100,200)
edge=cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)
cv2.imshow("Original | Canny",np.hstack((img,edge)))
cv2.waitKey(0);cv2.destroyAllWindows()
