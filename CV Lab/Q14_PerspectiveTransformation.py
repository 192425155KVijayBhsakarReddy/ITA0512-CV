import cv2,numpy as np
img=cv2.imread("images/input.jpg")
h,w=img.shape[:2]
p1=np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
p2=np.float32([[40,40],[w-40,0],[0,h-40],[w-40,h-40]])
M=cv2.getPerspectiveTransform(p1,p2)
out=cv2.warpPerspective(img,M,(w,h))
cv2.imshow("Original | Perspective",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
