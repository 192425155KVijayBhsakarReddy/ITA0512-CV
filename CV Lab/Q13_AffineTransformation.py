import cv2,numpy as np
img=cv2.imread("images/input.jpg")
h,w=img.shape[:2]
p1=np.float32([[50,50],[200,50],[50,200]])
p2=np.float32([[10,100],[200,50],[100,250]])
M=cv2.getAffineTransform(p1,p2)
out=cv2.warpAffine(img,M,(w,h))
cv2.imshow("Original | Affine",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
