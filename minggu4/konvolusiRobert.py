import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/home/rizki/wpcv/minggu4/bolaNew.png")
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# define kernels
Gx = np.array([[1, 0],
               [0, -1]], dtype=float)
Gy = np.array([[0, 1],
               [-1, 0]], dtype=float)
# Jika tidak punya scipy, implementasi sederhana (2x2 sliding window)

def roberts_numpy(image):
    h, w = image.shape
    Rx = np.zeros_like(image, dtype=float)
    Ry = np.zeros_like(image, dtype=float)
    # hitung untuk semua piksel kecuali batas kanan/bawah
    for i in range(h-1):
        for j in range(w-1):
            window = image[i:i+2, j:j+2].astype(float)
            Rx[i, j] = np.sum(window * Gx)
            Ry[i, j] = np.sum(window * Gy)
    magnitude = np.sqrt(Rx**2 + Ry**2)
    return magnitude, Rx, Ry

magnitude, Rx, Ry=roberts_numpy(grayscale)
plt.figure(figsize=(12.5,6))
clipped_mag = np.clip(magnitude, 0, 255).astype(np.uint8)

plt.subplot(1, 1, 1)
plt.imshow(clipped_mag, cmap='gray')
plt.title("Image_asli")
plt.axis("off")

plt.show()