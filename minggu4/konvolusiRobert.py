import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
image=cv2.cvtColor(cv2.imread("bola.png"), cv2.COLOR_BGR2GRAY)

# kernel robert
Gx = np.array([[ 0,  1],
               [-1,  0]], dtype=float)
Gy = np.array([[ 1,  0],
               [ 0, -1]], dtype=float)

# function for roberts convolution
def robertsConvolution(image, kernel):
    h, w = image.shape
    output = np.zeros_like(image, dtype=float)
    # hitung untuk semua piksel kecuali batas kanan/bawah
    for i in range(h-1):
        for j in range(w-1):
            window = image[i:i+2, j:j+2].astype(float)
            output[i, j] = np.sum(window * kernel)
    return output

# function to calculate gradient magnitude
Fx = robertsConvolution(image, Gx)
Fy = robertsConvolution(image, Gy)
finalOutput = np.sqrt(Fx**2+Fy**2)

# create window
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(finalOutput, cmap='gray')
plt.title("Edge Detection with roberts kernels")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(Fx, cmap='gray')
plt.title("convolusi roberts gradien X")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(Fy, cmap='gray')
plt.title("convolusi roberts gradien Y")
plt.axis("off")

plt.show()