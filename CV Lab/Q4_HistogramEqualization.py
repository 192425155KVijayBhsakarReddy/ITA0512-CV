import cv2,numpy as np
img=cv2.imread("images/input.jpg")
if img is None: raise FileNotFoundError("images/input.jpg not found")
g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
eq=cv2.equalizeHist(g)
g=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
eq=cv2.cvtColor(eq,cv2.COLOR_GRAY2BGR)
cv2.imshow("Gray | Equalized",np.hstack((g,eq)))
cv2.waitKey(0);cv2.destroyAllWindows()
