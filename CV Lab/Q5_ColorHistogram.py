import cv2
from matplotlib import pyplot as plt
img=cv2.imread("images/input.jpg")
if img is None: raise FileNotFoundError("images/input.jpg not found")
for i,c in enumerate(("b","g","r")):
    plt.plot(cv2.calcHist([img],[i],None,[256],[0,256]),color=c)
plt.title("Color Histogram")
plt.xlabel("Pixel Value"); plt.ylabel("Frequency"); plt.grid(); plt.show()
