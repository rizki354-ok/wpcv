# segmentasi citra dengan menggunakan method hough transform
# panggil image
# konversi ke grayscale
# lakukan percobaan:
#     pre-processing
#     1, menggunakan bluring citra
#     2. edge detection
# hipotesa: apakah img bluring bisa dilakkuan segmentasi dengan hough transform
# buat transformasi citra dengan hough transform
# tampilkan hasil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

img = cv2.imread("bola.png")
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12.5,6))

plt.subplot(1, 2, 1)
plt.imshow(grayscale, cmap='gray')
plt.title("Image_asli")
plt.axis("off")

plt.show()