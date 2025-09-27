import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# # --- Path file ---
# image_path = 
output_path = "matrix_outputRobert3.xlsx"

# --- Kernel ---
robert = np.array([[-1, 0],
                   [ 0, 1]], dtype=float)
otherKernel = np.array([[-0.5,0.2,0.1],
                        [0.2,0.1,0.4],
                        [0.3,-0.1,-0.2]], dtype=float)
blurKernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=float)


image_path = cv2.imread("/home/rizki/wpcv/minggu4/bola.png")
# image_path = cv2.imread("bolaNew.png")
image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

# --- Fungsi konvolusi manual ---
def convolution2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    pad = m // 2
    
    padded_img = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image, dtype=np.float32)
    
    kernel_sum = np.sum(kernel)
    for i in range(y):
        for j in range(x):
            region = padded_img[i:i+m, j:j+n]
            value = np.sum(region * kernel)
            if kernel_sum != 0:
                value /= kernel_sum  # normalisasi jika sum ≠ 0
            output[i, j] = value
    return output

# --- Proses konvolusi ---
output_matrix = convolution2d(image_path, blurKernel)

# --- Simpan ke Excel ---
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    pd.DataFrame(image_path).to_excel(writer, sheet_name="Matrix image", index=False, header=False)
    pd.DataFrame(otherKernel).to_excel(writer, sheet_name="Kernel", index=False, header=False)
    pd.DataFrame(output_matrix).to_excel(writer, sheet_name="Hasil Konvolusi", index=False, header=False)

# --- Visualisasi ---
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image_path, cmap="gray")
plt.title("Citra Asli (CSV)")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(output_matrix, cmap="gray")
plt.title("Hasil Konvolusi (Robert)")
plt.colorbar()

plt.tight_layout()
plt.show()
