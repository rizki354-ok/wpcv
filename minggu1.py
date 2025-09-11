import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar (BGR)
img = cv2.imread('apel.png')

# Konversi ke RGB (karena cv2 default-nya BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pisahkan channel R, G, B
R, G, B = cv2.split(img_rgb)

# Grayscale dengan rata-rata (R+G+B)/3
grayscale_avg = ((R.astype(np.float32) + G.astype(np.float32) + B.astype(np.float32)) / 3)
grayscale_avg = np.clip(grayscale_avg, 0, 255).astype(np.uint8)

# Grayscale dengan rumus 0.3R + 0.6G + 0.1B
grayscale_weighted = (0.3 * R.astype(np.float32) + 
                      0.6 * G.astype(np.float32) + 
                      0.1 * B.astype(np.float32))
grayscale_weighted = np.clip(grayscale_weighted, 0, 255).astype(np.uint8)

# Grayscale bawaan OpenCV
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tampilkan hasil
plt.figure(figsize=(12,8))

plt.subplot(2,4,1), plt.imshow(img_rgb), plt.title("RGB Image"), plt.axis("off")
plt.subplot(2,4,2), plt.imshow(R, cmap="gray"), plt.title("Red Channel"), plt.axis("off")
plt.subplot(2,4,3), plt.imshow(G, cmap="gray"), plt.title("Green Channel"), plt.axis("off")
plt.subplot(2,4,4), plt.imshow(B, cmap="gray"), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(2,4,5), plt.imshow(grayscale_avg, cmap="gray"), plt.title("Grayscale (R+G+B)/3"), plt.axis("off")
plt.subplot(2,4,6), plt.imshow(grayscale_weighted, cmap="gray"), plt.title("Grayscale (0.3R+0.6G+0.1B)"), plt.axis("off")
plt.subplot(2,4,7), plt.imshow(gray_cv, cmap="gray"), plt.title("OpenCV Grayscale"), plt.axis("off")

plt.tight_layout()
plt.show()
