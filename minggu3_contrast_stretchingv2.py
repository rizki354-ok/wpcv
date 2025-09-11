import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img_path = "apel.png"
img = cv2.imread(img_path)

# Grayscale dengan rumus luminosity
b, g, r = cv2.split(img)
grayscale = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

# Fungsi contrast stretching
def contrast_stretching(image):
    I_min = np.min(image)
    I_max = np.max(image)
    stretched = ((image - I_min) / (I_max - I_min) * 255).astype(np.uint8)
    return stretched

# Contrast stretching untuk citra berwarna
contrast_img = np.zeros_like(img)
for i in range(3):  # proses per channel B, G, R
    contrast_img[:, :, i] = contrast_stretching(img[:, :, i])

# Fungsi plotting histogram
def plot_histogram(image, color='k', title='Histogram'):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# Plot hasil
plt.figure(figsize=(14, 9))

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
plt.imshow(cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))
plt.title("Contrast Stretching (Color)")
plt.axis("off")

plt.subplot(2, 4, 6)
plot_histogram(contrast_img, color='k', title="Histogram Contrast Stretching")

plt.tight_layout()
plt.show()
