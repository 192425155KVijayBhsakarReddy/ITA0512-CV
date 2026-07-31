import cv2,numpy as np
img=cv2.imread("images/input.jpg")
wm=cv2.imread("images/watermark.png")
wm=cv2.resize(wm,(150,80))
out=img.copy()
roi=out[10:90,10:160]
out[10:90,10:160]=cv2.addWeighted(roi,0.7,wm,0.3,0)
cv2.imshow("Original | Watermark",np.hstack((img,out)))
cv2.waitKey(0);cv2.destroyAllWindows()
