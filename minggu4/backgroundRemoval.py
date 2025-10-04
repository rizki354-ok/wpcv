import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image
image = cv2.imread('bola.png')
original = image.copy()

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Canny edge detection
edges = cv2.Canny(gray, threshold1=50, threshold2=120)

# Step 4: Dilate edges to make them more solid
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
edges_dilated = cv2.dilate(edges, kernel, iterations=4)

# Step 5: Find contours
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Create mask from contours
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Optional: Smooth the mask
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Step 7: Convert mask to 3-channel
mask_3ch = cv2.merge([mask, mask, mask])

# Step 8: Apply mask to original image
foreground = cv2.bitwise_and(original, mask_3ch)

# Step 9: Replace background (optional)
background_color = (255, 255, 255)  # white background
background = np.full_like(original, background_color)
inverted_mask = cv2.bitwise_not(mask_3ch)
final = cv2.bitwise_or(foreground, cv2.bitwise_and(background, inverted_mask))

# Show result
plt.figure()

plt.subplot(2, 6, 1)
plt.title('citra asli')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 6, 2)
plt.title('edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 6, 3)
plt.title('edges_dilated')
plt.imshow(edges_dilated, cmap='gray')
plt.axis('off')

# plt.subplot(2, 6, 4)
# plt.title('citra edge detection')
# plt.imshow(contours, cmap='gray')
# plt.axis('off')

plt.subplot(2, 6, 5)
plt.title('mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 6, 6)
plt.title('mask_3ch')
plt.imshow(mask_3ch, cmap='gray')
plt.axis('off')

plt.subplot(2, 6, 7)
plt.title('foreground')
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.axis('off')

# plt.subplot(2, 6, 8)
# plt.title('citra edge detection')
# plt.imshow(cv2.cvtColor(background_color, cv2.COLOR_BGR2RGB))
# plt.axis('off')

plt.subplot(2, 6, 9)
plt.title('background')
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 6, 10)
plt.title('inverted_mask')
plt.imshow(cv2.cvtColor(inverted_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 6, 11)
plt.title('final')
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.axis('off')


# plt.subplot(2, 6, 3)
# plt.title('citra advanced edge detection')
# plt.imshow(cv2.cvtColor(contours, cv2.COLOR_BGR2RGB))
# plt.axis('off')

plt.show()