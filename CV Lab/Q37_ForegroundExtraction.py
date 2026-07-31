import cv2,numpy as np
img=cv2.imread("images/input.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv,(0,40,40),(180,255,255))
fg=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("Original | Foreground",np.hstack((img,fg)))
cv2.waitKey(0);cv2.destroyAllWindows()
