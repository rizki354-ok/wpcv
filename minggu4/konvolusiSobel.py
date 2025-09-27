import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar & ubah ke grayscale
img = cv2.imread("/home/rizki/wpcv/minggu4/bolaNew.png")
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Kernel Sobel (3x3)
Gx = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=float)

Gy = np.array([[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]], dtype=float)

# 3. Implementasi Sobel manual dengan sliding window 3x3
def sobel_numpy(image):
    h, w = image.shape
    Rx = np.zeros_like(image, dtype=float)
    Ry = np.zeros_like(image, dtype=float)

    # loop semua piksel kecuali border (karena kernel 3x3)
    for i in range(1, h-1):
        for j in range(1, w-1):
            window = image[i-1:i+2, j-1:j+2].astype(float)
            Rx[i, j] = np.sum(window * Gx)
            Ry[i, j] = np.sum(window * Gy)

    magnitude = np.sqrt(Rx**2 + Ry**2)
    return magnitude, Rx, Ry

# 4. Jalankan sobel
magnitude, Rx, Ry = sobel_numpy(grayscale)

# 5. Clipping supaya muat ke 0–255
clipped_mag = np.clip(magnitude, 0, 255).astype(np.uint8)

# 6. Tampilkan hasil
plt.figure(figsize=(15,6))

plt.subplot(1,1,1)
plt.imshow(clipped_mag, cmap='gray')
plt.title("Sobel Magnitude")
plt.axis("off")

plt.show()
