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
def convolution2d(image, kernel, normalize='sum'):
    """
    Convolve a 2D image with a kernel, then divide each sum by the kernel sum.
    normalize: 'sum' -> divide by kernel.sum()
               'abs' -> divide by np.abs(kernel).sum()
               None  -> no normalization (divide by 1)
    """
    m, n = kernel.shape
    if m != n or (m % 2) == 0:
        warnings.warn("Fungsi ini mengasumsikan kernel persegi ganjil. Pastikan ukuran kernel ganjil.")
    pad = m // 2

    padded_img = np.pad(image, pad, mode='constant', constant_values=0)
    y, x = image.shape
    output = np.zeros_like(image, dtype=np.float32)

    if normalize == 'sum':
        denom = kernel.sum()
    elif normalize == 'abs':
        denom = np.abs(kernel).sum()
    else:
        denom = 1.0

    if abs(denom) < 1e-12:
        warnings.warn("Jumlah kernel hampir atau sama dengan nol. Menghindari pembagian nol -> tidak melakukan normalisasi.")
        denom = 1.0

    for i in range(y):
        for j in range(x):
            region = padded_img[i:i+m, j:j+n]
            conv_sum = np.sum(region * kernel)
            output[i, j] = conv_sum / denom

    return output
# --- Proses konvolusi ---
output_matrix = convolution2d(image_matrix, kernel)
channel_gambar = pd.DataFrame(image_matrix)
channel_kernel = pd.DataFrame(kernel)
channel_output = pd.DataFrame(output_matrix)

with pd.ExcelWriter("hasil.xlsx") as writer:
    channel_gambar.to_excel(writer, sheet_name="Matriks Gambar", index=False, header=False)
    channel_kernel.to_excel(writer, sheet_name="Kernel", index=False, header=False)
    channel_output.to_excel(writer, sheet_name="Hasil", index=False, header=False)

# --- Tampilkan hasil ---
plt.figure(figsize=(10,4))

# Gambar asli
plt.subplot(1,2,1)
plt.imshow(image_matrix, cmap="gray")
plt.title("Citra Asli")
plt.colorbar()

# Hasil konvolusi
plt.subplot(1,2,2)
plt.imshow(output_matrix, cmap="gray")
plt.title("Hasil Konvolusi")
plt.colorbar()

print("Matriks citra:")
print(image_matrix)
print("\nKernel:")
print(kernel)

plt.show()