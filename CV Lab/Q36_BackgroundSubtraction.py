import cv2
bs=cv2.createBackgroundSubtractorMOG2()
cap=cv2.VideoCapture("images/video.mp4")
while True:
    r,f=cap.read()
    if not r: break
    mask=bs.apply(f)
    cv2.imshow("Original",f)
    cv2.imshow("Background Subtraction",mask)
    if cv2.waitKey(30)&0xFF==27: break
cap.release();cv2.destroyAllWindows()
