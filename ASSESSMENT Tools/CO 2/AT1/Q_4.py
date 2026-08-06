import cv2
import matplotlib.pyplot as plt

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "image.jpg")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not read image at '{image_path}'. Check if file exists.")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.hist(img.ravel(), bins=256, range=[0,256], color='black')
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.show()