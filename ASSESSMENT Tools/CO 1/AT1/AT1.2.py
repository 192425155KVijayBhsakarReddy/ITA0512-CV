import cv2
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("image2.jpg", 0)

# Apply thermal color map
thermal = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB))
plt.title("Thermal Image")
plt.axis("off")

plt.show()
