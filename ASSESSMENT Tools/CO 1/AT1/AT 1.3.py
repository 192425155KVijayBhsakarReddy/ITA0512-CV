import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("image3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create motion blur kernel
size = 15
kernel = np.zeros((size, size))
kernel[int((size-1)/2), :] = np.ones(size)
kernel = kernel / size

# Apply blur
motion_blur = cv2.filter2D(img, -1, kernel)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(motion_blur)
plt.title("Motion Blur")
plt.axis("off")

plt.show()
