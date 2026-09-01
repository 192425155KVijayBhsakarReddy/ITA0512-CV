
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
noise = np.random.normal(0, 25, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
mean = cv2.blur(noisy, (5, 5))
median = cv2.medianBlur(noisy, 5)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
nlm = cv2.fastNlMeansDenoisingColored(
    noisy, None, 10, 10, 7, 21
)
images = [img, noisy, mean, median, gaussian, bilateral, nlm]
titles = [
    "Original",
    "Noisy Image",
    "Mean Filter",
    "Median Filter",
    "Gaussian Filter",
    "Bilateral Filter",
    "Non-Local Means"
]

plt.figure(figsize=(14, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
