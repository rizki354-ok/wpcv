# membaca gambar dan menampilkan histogramnya
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("data gambar M3/Data Image/kabut.jpeg")
b, g, r = cv2.split(img) 
img_gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
min_val = np.min(img_gray)
max_val = np.max(img_gray)
stretched_grayscale = ((img_gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
min_val = np.min(img)
max_val = np.max(img)
stretched_rgb_direct = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
stretched_rgb_manual = np.zeros_like(img)
min_val = np.min(stretched_rgb_manual[:, :, 0])
max_val = np.max(stretched_rgb_manual[:, :, 0])
stretched_rgb_manual[:, :, 0] = ((stretched_rgb_manual[:, :, 0] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
min_val = np.min(stretched_rgb_manual[:, :, 1])
max_val = np.max(stretched_rgb_manual[:, :, 1])
stretched_rgb_manual[:, :, 1] = ((stretched_rgb_manual[:, :, 1] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
min_val = np.min(stretched_rgb_manual[:, :, 2])
max_val = np.max(stretched_rgb_manual[:, :, 2])
stretched_rgb_manual[:, :, 2] = ((stretched_rgb_manual[:, :, 2] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

plt.figure(figsize=(12.5,6))

plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("1")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.hist(img.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 3)
plt.imshow(cv2.cvtColor(stretched_grayscale, cv2.COLOR_BGR2RGB))
plt.title("2")
plt.axis("off")

plt.subplot(3, 4, 4)
plt.hist(stretched_grayscale.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(stretched_rgb_direct, cv2.COLOR_BGR2RGB))
plt.title("3")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.hist(stretched_rgb_direct.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 7)
plt.imshow(cv2.cvtColor(stretched_rgb_manual, cv2.COLOR_BGR2RGB))
plt.title("4")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.hist(stretched_rgb_manual.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
plt.title("Histogram stretching citra Grayscale")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")



plt.show()