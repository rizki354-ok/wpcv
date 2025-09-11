# 5_hist_equal_spec.py
# Histogram equalization (manual) dan histogram specification (manual)
# Gunakan grayscale terlebih dahulu (luminosity)
# Tampilkan original, equalized, specified + histogram masing-masing

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "apel.png"

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Tidak dapat menemukan file: {img_path}")

b, g, r = cv2.split(img)
gray = (0.3 * r + 0.6 * g + 0.1 * b).round().astype(np.uint8)

# --- Histogram Equalization (manual) ---
hist, bins = np.histogram(gray.flatten(), bins=256, range=(0,255))
pdf = hist.astype(np.float64) / hist.sum()
cdf = pdf.cumsum()
map_eq = np.floor(255 * cdf).astype(np.uint8)   # mapping 0..255

equalized = map_eq[gray]   # apply mapping

# --- Histogram Specification (matching) ---
# Kita buat target PDF secara programatik (contoh: Gaussian tertempat)
x = np.arange(256)
mu = 150.0
sigma = 30.0
target_pdf = np.exp(-0.5 * ((x - mu)/sigma)**2)
target_pdf = target_pdf / target_pdf.sum()
target_cdf = target_pdf.cumsum()

# Sumber cdf (dari grayscale asli)
src_hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
src_pdf = src_hist.astype(np.float64) / src_hist.sum()
src_cdf = src_pdf.cumsum()

# mapping spesifikasi: untuk setiap level r cari s sehingga src_cdf[r] ~ target_cdf[s]
map_spec = np.zeros(256, dtype=np.uint8)
for r in range(256):
    # cari s minimal dimana target_cdf[s] >= src_cdf[r]
    s = np.searchsorted(target_cdf, src_cdf[r])
    if s > 255:
        s = 255
    map_spec[r] = s

specified = map_spec[gray]

# --- histogram data ---
bins_centers = (bins[:-1] + bins[1:]) / 2
hist_orig, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
hist_eq, _ = np.histogram(equalized.flatten(), bins=256, range=(0,255))
hist_spec, _ = np.histogram(specified.flatten(), bins=256, range=(0,255))

# --- tampilkan ---
plt.figure(figsize=(12,8))

plt.subplot(3,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis('off')

plt.subplot(3,3,2)
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')

plt.subplot(3,3,3)
plt.imshow(specified, cmap='gray')
plt.title("Histogram Specification (target Gaussian)")
plt.axis('off')

plt.subplot(3,3,4)
plt.plot(bins_centers, hist_orig)
plt.title("Histogram Original")
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

plt.subplot(3,3,5)
plt.plot(bins_centers, hist_eq)
plt.title("Histogram Equalized")
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

plt.subplot(3,3,6)
plt.plot(bins_centers, hist_spec)
plt.title("Histogram Specified")
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

# tampilkan juga PDF target sebagai referensi
plt.subplot(3,3,8)
plt.plot(x, target_pdf)
plt.title("Target PDF (normalisasi)")
plt.xlabel("Intensitas")
plt.ylabel("Prob.")

plt.tight_layout()
plt.show()

cv2.imwrite("hist_equalized.png", equalized)
cv2.imwrite("hist_specified.png", specified)
print("Selesai: 'hist_equalized.png' dan 'hist_specified.png' tersimpan.")
