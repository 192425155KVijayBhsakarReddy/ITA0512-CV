import cv2
import time
import math

cap=cv2.VideoCapture(r"P:\SSE\CV\ASSESSMENT Tools\CO 5\AT 1\CCTV_VIDEO.mp4")

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

back_sub=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=50,detectShadows=True)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

MIN_CONTOUR_AREA=500
MOTION_CONFIRM_FRAMES=4
SUDDEN_MOVEMENT_THRESHOLD=150
LOITERING_TIME=10
motion_counter=0
previous_center=None
object_start_time=None
alert_cooldown=5
last_alert_time=0

def generate_alert(alert_type):
    global last_alert_time
    current_time=time.time()
    if current_time-last_alert_time>=alert_cooldown:
        print("================================")
        print("ALERT:",alert_type)
        print("Time:",time.strftime("%H:%M:%S"))
        print("================================")
        filename="alert_"+str(int(current_time))+".jpg"
        cv2.imwrite(filename,frame)
        last_alert_time=current_time

while True:
    ret,frame=cap.read()
    if not ret:
        print("Video ended or frame could not be read.")
        break

    frame=cv2.resize(frame,(640,360))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    mask=back_sub.apply(gray)
    _,mask=cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    motion_detected=False
    current_center=None
    largest_area=0
    largest_box=None

    for contour in contours:
        area=cv2.contourArea(contour)
        if area<MIN_CONTOUR_AREA:
            continue
        motion_detected=True
        x,y,w,h=cv2.boundingRect(contour)
        if area>largest_area:
            largest_area=area
            largest_box=(x,y,w,h)
            center_x=x+w//2
            center_y=y+h//2
            current_center=(center_x,center_y)

    if largest_box is not None:
        x,y,w,h=largest_box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(frame,current_center,5,(255,0,0),-1)

    if motion_detected:
        motion_counter+=1
    else:
        motion_counter=0

    if motion_counter>=MOTION_CONFIRM_FRAMES:
        cv2.putText(frame,"MOTION CONFIRMED",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        generate_alert("Persistent Motion Detected")

    if current_center is not None:
        if previous_center is None:
            previous_center=current_center
            object_start_time=time.time()
        else:
            dx=current_center[0]-previous_center[0]
            dy=current_center[1]-previous_center[1]
            distance=math.sqrt(dx*dx+dy*dy)
            dt=1/30
            velocity=distance/dt

            if velocity>SUDDEN_MOVEMENT_THRESHOLD:
                cv2.putText(frame,"SUDDEN MOVEMENT",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                generate_alert("Sudden Movement Detected")

            previous_center=current_center
    else:
        previous_center=None
        object_start_time=None

    if current_center is not None and object_start_time is not None:
        elapsed_time=time.time()-object_start_time
        if elapsed_time>=LOITERING_TIME:
            cv2.putText(frame,"LOITERING DETECTED",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            generate_alert("Possible Loitering")

    roi_x1=200
    roi_y1=100
    roi_x2=600
    roi_y2=350

    cv2.rectangle(frame,(roi_x1,roi_y1),(roi_x2,roi_y2),(255,0,255),2)
    cv2.putText(frame,"RESTRICTED AREA",(roi_x1,roi_y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

    if current_center is not None:
        cx,cy=current_center
        if roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2:
            cv2.putText(frame,"INTRUSION DETECTED",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            generate_alert("Restricted Area Intrusion")

    status="MOTION" if motion_detected else "NO MOTION"

    cv2.putText(frame,"Status: "+status,(10,350),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.imshow("Automated Video Surveillance",frame)
    cv2.imshow("Motion Mask",mask)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Surveillance system stopped.")