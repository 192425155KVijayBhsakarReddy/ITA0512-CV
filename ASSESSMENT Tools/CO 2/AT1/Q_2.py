import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "image.jpg")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not read image at '{image_path}'. Check if file exists.")

# Convert to float
img_float = img.astype(np.float32)

# Log Transformation
c = 255 / np.log(1 + np.max(img_float))
log_image = c * np.log(1 + img_float)

log_image = np.array(log_image, dtype=np.uint8)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(log_image, cmap='gray')
plt.title("Log Transform")
plt.axis("off")

plt.show()