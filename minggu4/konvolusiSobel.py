# Konvolusi dengan kernel sobel dengan gradien X dan Gradien Y untuk mendeteksi tepi
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read a image
img=cv2.cvtColor(cv2.imread("bola.png"), cv2.COLOR_BGR2GRAY)

# kernel sobel
Gy = np.array([[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]], dtype=float)
Gx = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=float)

# fungsi konvolusi dasar manual
def convolution2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    pad = m // 2

    # menambahkan paddingdengan angka 0 agar hasil tepi bisa dihitung juga dan menambahkan variabel output yang menyimpan hasil konvolusi
    paddingImg = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image, dtype=float)

    for i in range(y):
        for j in range(x):
            region = paddingImg[i:i+m, j:j+n]
            output[i, j] = np.sum(region * kernel)
    return output

# konvolusi Sobel kernel for edge detection
def convolutionSobel(image, Rx, Ry):
    sobelX = convolution2d(image, Rx)    
    sobelY = convolution2d(image, Ry)
    magnitude = np.sqrt(sobelX**2 + sobelY**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    return magnitude, sobelX, sobelY

SobelConvolution, Gx, Gy = convolutionSobel(img, Gx, Gy)

# create window
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("gambar")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(SobelConvolution, cmap='gray')
plt.title("hasil convolusi sobel")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(Gx, cmap='gray')
plt.title("convolusi sobel gradien X")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(Gy, cmap='gray')
plt.title("convolusi sobel gradien Y")
plt.axis("off")

plt.show()