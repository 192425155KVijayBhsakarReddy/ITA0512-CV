import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("input.jpg")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Median Filter to remove background noise
filtered = cv2.medianBlur(image_rgb, 5)

# Convert to grayscale
gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

# Segment using Otsu Thresholding
_, segmented = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(filtered)
plt.title("After Median Filter")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(segmented, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()
