import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img_path = "apel.png"
img = cv2.imread(img_path)
b, g, r = cv2.split(img)  # Split channel

# Grayscale: rata-rata
gray_avg = ((r.astype(np.float32) + g + b) / 3.0).astype(np.uint8)

# Grayscale: luminosity
gray_lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

# Fungsi untuk plotting histogram
def plot_histogram(image, color='k', title='Histogram'):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# Plot hasil
plt.figure(figsize=(15, 12))

# 1. Citra asli
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(3, 4, 2)
plot_histogram(img, color='k', title="Histogram Citra Asli")

# 2. Channel R
plt.subplot(3, 4, 3)
plt.imshow(r, cmap='Reds')
plt.title("Channel R")
plt.axis("off")

plt.subplot(3, 4, 4)
plot_histogram(r, color='r', title="Histogram R")

# 3. Channel G
plt.subplot(3, 4, 5)
plt.imshow(g, cmap='Greens')
plt.title("Channel G")
plt.axis("off")

plt.subplot(3, 4, 6)
plot_histogram(g, color='g', title="Histogram G")

# 4. Channel B
plt.subplot(3, 4, 7)
plt.imshow(b, cmap='Blues')
plt.title("Channel B")
plt.axis("off")

plt.subplot(3, 4, 8)
plot_histogram(b, color='b', title="Histogram B")

# 5. Grayscale Average
plt.subplot(3, 4, 9)
plt.imshow(gray_avg, cmap='gray')
plt.title("Grayscale (Average)")
plt.axis("off")

plt.subplot(3, 4, 10)
plot_histogram(gray_avg, color='gray', title="Histogram Grayscale (Average)")

# 6. Grayscale Luminosity
plt.subplot(3, 4, 11)
plt.imshow(gray_lum, cmap='gray')
plt.title("Grayscale (Luminosity)")
plt.axis("off")

plt.subplot(3, 4, 12)
plot_histogram(gray_lum, color='gray', title="Histogram Grayscale (Luminosity)")

plt.tight_layout()
plt.show()
