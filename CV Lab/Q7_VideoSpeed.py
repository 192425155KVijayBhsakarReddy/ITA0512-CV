import cv2
cap=cv2.VideoCapture("images/video.mp4")
if not cap.isOpened(): raise FileNotFoundError("images/video.mp4 not found")
while True:
    ret,frame=cap.read()
    if not ret: break
    cv2.imshow("Video - Slow Motion",frame)
    if cv2.waitKey(80)&0xFF==27: break
cap.release();cv2.destroyAllWindows()
