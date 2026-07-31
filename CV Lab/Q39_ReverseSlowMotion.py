import cv2
cap=cv2.VideoCapture("images/video.mp4")
frames=[]
while True:
    r,f=cap.read()
    if not r: break
    frames.append(f)
cap.release()
for f in frames[::-1]:
    cv2.imshow("Reverse Slow Motion",f)
    if cv2.waitKey(100)&0xFF==27: break
cv2.destroyAllWindows()
