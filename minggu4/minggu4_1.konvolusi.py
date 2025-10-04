# Dasar konvolusi dengan menggunakan matrix image buatan dan kernel buatan
# dihitung dengan rumus dasar konvolusi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Matriks citra ---
image_matrix = np.array([
    [62, 195, 179, 179, 179, 179, 179, 179, 166, 109],
    [127, 133, 92, 91, 91, 90, 89, 89, 79, -9.7],
    [97, 82, 60, 55, 49, 44, 38, 33, 76, 14],
    [83, 64, 115, 111, 95, 83, 81, 45, 75, 84],
    [126, 87, 111, 85, 27, 3.5, 7.1, 13, 43, 84],
    [119, 143, 76, 10, -4.1, 11, 8.9, 70, 119, 63],
    [117, 59, 38, -47, 26, 25, 13, 74, 127, 28],
    [123, 113, 57, 63, 46, 35, 33, 110, 162, 28],
    [124, 108, 90, 138, 117, 131, 131, 115, 187, 74],
    [204, 126, 127, 128, 129, 130, 126, 114, 143, 63]
], dtype=float)

# --- Kernel ---
kernel = np.array([
    [-0.5, 0.2, 0.1],
    [0.2, 0.1, 0.4],
    [0.3, -0.1, -0.2]
], dtype=float)

# fungsi konvolusi manual
def convolution2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    pad = m // 2

    # menambahkan paddingdengan angka 0 agar hasil tepi bisa dihitung juga dan menambahkan variabel output yang menyimpan hasil konvolusi
    paddingImg = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for i in range(y):
        for j in range(x):
            region = paddingImg[i:i+m, j:j+n]
            output[i, j] = np.sum(region * kernel)
    return output

# lakukan konvolusi
convolusi_img = convolution2d(image_matrix, kernel)

# convert to csv
channelGambar = pd.DataFrame(image_matrix)
channelKernel = pd.DataFrame(kernel)
channelHasil = pd.DataFrame(convolusi_img)

with pd.ExcelWriter("hasilConvolusiDasar(custom konvolusi).xlsx") as writer:
    channelGambar.to_excel(writer, sheet_name="Matriks Gambar", index=False, header=False)
    channelKernel.to_excel(writer, sheet_name="Kernel", index=False, header=False)
    channelHasil.to_excel(writer, sheet_name="Hasil", index=False, header=False)

# full
# plt.figure(figsize=(13.55, 6.7))
# auto
plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(image_matrix, cmap="gray")
plt.title("image_matrix (asli)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(convolusi_img, cmap="gray")
plt.title("Konvolusi_image_matrix (Konvolusi)")
plt.axis("off")

plt.tight_layout()
plt.show()