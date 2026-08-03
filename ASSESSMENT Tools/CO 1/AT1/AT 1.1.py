import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reduce intensity levels (Banding)
levels = 8
banded = (img // (256 // levels)) * (256 // levels)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(banded)
plt.title("Banding Effect")
plt.axis("off")

plt.show()
