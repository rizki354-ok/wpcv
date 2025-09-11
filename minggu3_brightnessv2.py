import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img_path = "apel.png"
img = cv2.imread(img_path)

# Grayscale dengan rumus luminosity
b, g, r = cv2.split(img)
grayscale = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

# Fungsi untuk brightness (bisa negatif atau positif)
def brightness_adjust(image, k):
    # konversi ke int32 agar tidak overflow, lalu clip ke [0,255]
    bright = np.clip(image.astype(np.int32) + k, 0, 255)
    return bright.astype(np.uint8)

# Ubah nilai k di sini
k_value = -80   # negatif = lebih gelap, positif = lebih terang
bright_img = brightness_adjust(img, k_value)

# Fungsi untuk histogram
def plot_histogram(image, title="Histogram"):
    colors = ('b', 'g', 'r')
    if len(image.shape) == 2:  # grayscale
        plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
    else:  # warna
        for i, col in enumerate(colors):
            plt.hist(image[:,:,i].ravel(), bins=256, range=(0, 256), color=col, alpha=0.6)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# Plot hasil
plt.figure(figsize=(14, 10))

# 1. Citra Asli
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(3, 2, 2)
plot_histogram(img, title="Histogram Citra Asli")

# 2. Grayscale Luminosity
plt.subplot(3, 2, 3)
plt.imshow(grayscale, cmap='gray')
plt.title("Grayscale (Luminosity)")
plt.axis("off")

plt.subplot(3, 2, 4)
plot_histogram(grayscale, title="Histogram Grayscale")

# 3. Brightness
plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB))
plt.title(f"Brightness (k={k_value})")
plt.axis("off")

plt.subplot(3, 2, 6)
plot_histogram(bright_img, title=f"Histogram Brightness (k={k_value})")

plt.tight_layout()
plt.show()