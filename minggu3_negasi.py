import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img_path = "apel.png"
img = cv2.imread(img_path)
b, g, r = cv2.split(img)

# Grayscale dengan rumus luminosity
grayscale = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

# Fungsi untuk membuat citra negasi
def negative_image(image, k=255):
    return (k - image).astype(np.uint8)

# Negasi dari grayscale
k_value = 255   # bisa diubah
negasi = negative_image(grayscale, k=k_value)

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

# 3. Negasi + histogram
plt.subplot(2, 4, 5)
plt.imshow(negasi, cmap='gray')
plt.title(f"Negasi (k={k_value})")
plt.axis("off")

plt.subplot(2, 4, 6)
plot_histogram(negasi, color='gray', title=f"Histogram Negasi (k={k_value})")

plt.tight_layout()
plt.show()
