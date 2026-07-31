import cv2,numpy as np
text=input("Enter text: ")
img=np.ones((300,700,3),dtype=np.uint8)*255
cv2.putText(img,text,(30,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
cv2.imshow("Text",img)
cv2.waitKey(0);cv2.destroyAllWindows()
