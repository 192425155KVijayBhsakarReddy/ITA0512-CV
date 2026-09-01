import os

video_path=r"P:\SSE\CV\ASSESSMENT Tools\CO 5\AT 1\CCTV_VIDEO.mp4"

print("Video exists:",os.path.exists(video_path))

cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()