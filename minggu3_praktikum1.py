# membaca gambar dan menampilkan histogramnya
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("data gambar M3/Data Image/kabut.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
min_val = np.min(img)
max_val = np.max(img)
stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

plt.figure(figsize=(12.5,6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB))
plt.title("Citra setekah contras sctretching")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.hist(stretched.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
plt.title("Histogram contrast stretching")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")



plt.show()