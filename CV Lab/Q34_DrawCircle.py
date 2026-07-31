import cv2,numpy as np
img=np.ones((400,400,3),dtype=np.uint8)*255
cv2.circle(img,(200,200),100,(0,255,0),3)
cv2.imshow("Circle",img)
cv2.waitKey(0);cv2.destroyAllWindows()
