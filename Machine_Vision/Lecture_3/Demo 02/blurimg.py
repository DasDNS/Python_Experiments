import cv2
import numpy as np

# Read the image
image = cv2.imread('Lenna.png')

# Check if image was loaded successfully
if image is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# 1. Average Blur (Box Filter)
average_blur = cv2.blur(image, (5, 5))

# 2. Gaussian Blur (most common, better quality)
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# 3. Median Blur (good for salt-and-pepper noise)
median_blur = cv2.medianBlur(image, 15)

# 4. Bilateral Filter (preserves edges while blurring)
bilateral_blur = cv2.bilateralFilter(image, 15, 75, 75)

# Display all results
cv2.imshow('Original Image', image)
cv2.imshow('Average Blur', average_blur)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)
cv2.imshow('Bilateral Filter', bilateral_blur)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()