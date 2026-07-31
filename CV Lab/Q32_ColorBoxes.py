import cv2,numpy as np
h=int(input("Height: ")); w=int(input("Width: "))
img=np.ones((h,w,3),dtype=np.uint8)*255
s=min(h,w)//10
img[:s,:s]=[0,0,0]
img[:s,-s:]=[255,0,0]
img[-s:,:s]=[0,255,0]
img[-s:,-s:]=[0,0,255]
cv2.imshow("Colored Boxes",img)
cv2.waitKey(0);cv2.destroyAllWindows()
