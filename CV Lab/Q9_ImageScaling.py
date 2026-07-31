import cv2,numpy as np
img=cv2.imread("images/input.jpg")
big=cv2.resize(img,(img.shape[1],img.shape[0]),fx=0,fy=0)
big=cv2.resize(img,None,fx=2,fy=2)
big=cv2.resize(big,(img.shape[1],img.shape[0]))
small=cv2.resize(img,None,fx=0.5,fy=0.5)
small=cv2.resize(small,(img.shape[1],img.shape[0]))
cv2.imshow("Original | Bigger | Smaller",np.hstack((img,big,small)))
cv2.waitKey(0);cv2.destroyAllWindows()
