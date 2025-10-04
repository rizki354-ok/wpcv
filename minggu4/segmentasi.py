import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bola.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

gray_blur = cv2.medianBlur(gray, 5)

binary_mask = np.zeros_like(gray_blur)

circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=300,
    param1=20,
    param2=60,
    minRadius=20,
    maxRadius=100
)

output = edges.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (255, 0, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 255, 0), 3)
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(img.shape[1], x+r), min(img.shape[0], y+r)
        bola_crop = img_rgb[y1:y2, x1:x2]
        
else:
    print('Bola tidak ditemukan')

plt.figure(figsize=(18, 8))

plt.subplot(1, 5, 1)
plt.imshow(img_rgb)
plt.title('Citra asli')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(gray, cmap='gray')
plt.title('Citra Gray')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(output,cmap='gray')
plt.title('HoughCircles Output')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(bola_crop,cmap='gray')
plt.title('HoughCircles Output')
plt.axis('off')

plt.tight_layout()
plt.show()
