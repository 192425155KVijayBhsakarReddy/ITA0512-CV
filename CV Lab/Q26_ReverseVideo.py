import cv2, numpy as np
cap=cv2.VideoCapture("images/video.mp4")
frames=[]
while True:
    r,f=cap.read()
    if not r: break
    frames.append(f)
cap.release()
for f in frames[::-1]:
    cv2.imshow("Reverse Video",f)
    if cv2.waitKey(30)&0xFF==27: break
cv2.destroyAllWindows()
