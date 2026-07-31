import cv2,pytesseract
cap=cv2.VideoCapture("images/video.mp4")
while True:
    r,f=cap.read()
    if not r: break
    text=pytesseract.image_to_string(f)
    print(text.strip())
    cv2.imshow("Video",f)
    if cv2.waitKey(30)&0xFF==27: break
cap.release();cv2.destroyAllWindows()
