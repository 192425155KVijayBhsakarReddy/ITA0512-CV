import cv2,numpy as np
img=np.ones((400,400,3),dtype=np.uint8)*255
cv2.rectangle(img,(50,50),(350,300),(255,0,0),3)
cv2.imshow("Rectangle",img)
cv2.waitKey(0);cv2.destroyAllWindows()
