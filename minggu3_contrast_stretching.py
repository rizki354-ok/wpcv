import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img_path = "apel.png"
img = cv2.imread(img_path)
b, g, r = cv2.split(img)

# Grayscale dengan rumus luminosity
grayscale = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

# Fungsi untuk contrast stretching
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Hasil contrast stretching
contrast_img = contrast_stretching(grayscale)

# Fungsi plotting histogram
def plot_histogram(image, color='k', title='Histogram'):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# Plot hasil
plt.figure(figsize=(14, 8))

# 1. Citra asli + histogram
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 4, 2)
plot_histogram(img, color='k', title="Histogram Citra Asli")

# 2. Grayscale (Luminosity) + histogram
plt.subplot(2, 4, 3)
plt.imshow(grayscale, cmap='gray')
plt.title("Grayscale (Luminosity)")
plt.axis("off")

plt.subplot(2, 4, 4)
plot_histogram(grayscale, color='gray', title="Histogram Grayscale")

# 3. Contrast Stretching + histogram
plt.subplot(2, 4, 5)
plt.imshow(contrast_img, cmap='gray')
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(2, 4, 6)
plot_histogram(contrast_img, color='gray', title="Histogram Contrast Stretching")

plt.tight_layout()
plt.show()
