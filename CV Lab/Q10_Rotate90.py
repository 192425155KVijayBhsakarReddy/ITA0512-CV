import cv2,numpy as np
img=cv2.imread("images/input.jpg")
rot=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
rot=cv2.resize(rot,(img.shape[1],img.shape[0]))
cv2.imshow("Original | Rotate 90",np.hstack((img,rot)))
cv2.waitKey(0);cv2.destroyAllWindows()
