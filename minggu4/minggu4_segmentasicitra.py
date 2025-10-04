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

import cv2
import numpy as np
import matplotlib.pyplot as plt

# image edge detection
imageGray = cv2.cvtColor(cv2.imread("bola.png"), cv2.COLOR_BGR2GRAY)
# bluring image
imageBluring = cv2.medianBlur(imageGray, 5)
# edge detecion
imageEdgeDetection = cv2.Canny(imageGray, 100, 200)
binary_mask = np.zeros_like(imageGray)

# Deteksi lingkaran dengan Hough Transform
circles = cv2.HoughCircles(
    imageEdgeDetection,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=300,
    param1=20,
    param2=60,
    minRadius=20,
    maxRadius=100
)
output = binary_mask.copy()

# v1
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        # Gambar lingkaran
        cv2.circle(output, (x, y), r, (255, 0, 0), 3)
        # Gambar titik pusat
        cv2.circle(output, (x, y), 2, (0, 255, 0), 3)

# v2
if circles is not None:
    circles = np.uint16(np.around(circles))
    for(x, y, r) in circles[0, :1]:
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(imageGray.shape[1], x+r), min(imageGray.shape[0], y+r)
        bola_crop = imageGray[y1:y2, x1:x2]
        

# create window
plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(imageGray, cmap='gray')
plt.title('Citra asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output, cmap='gray')
plt.title('Citra asli')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(bola_crop, cmap='gray')
plt.title('Citra asli')
plt.axis('off')

plt.show()