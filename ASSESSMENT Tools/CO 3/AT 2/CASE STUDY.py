import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("input.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
harris = cv2.cornerHarris(
    np.float32(gray),
    2,      # block size
    3,      # Sobel kernel
    0.04    # Harris parameter
)
harris = cv2.dilate(harris, None)
harris_img = img.copy()
harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]

corners = cv2.goodFeaturesToTrack(
    gray,
    100,    # maximum corners
    0.01,   # quality level
    10      # minimum distance
)
shi_img = img.copy()
if corners is not None:
    corners = np.int32(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(shi_img, (x, y), 5, (0, 255, 0), -1)

plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(shi_img, cv2.COLOR_BGR2RGB))
plt.title("Shi-Tomasi Corner Detection")
plt.axis("off")
plt.tight_layout()
plt.show()
