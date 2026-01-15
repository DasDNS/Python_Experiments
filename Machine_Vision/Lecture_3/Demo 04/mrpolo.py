import cv2
import numpy as np

# Read the image
image = cv2.imread('Lenna.png')

# Check if image was loaded successfully
if image is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary threshold to get a binary image
# Morphological operations work best on binary images
#Grey value between 127 and 255 is considered as 1
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Define the structuring element (kernel)
# You can use different shapes: MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS
kernel_size = (5, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

# Alternative kernels:
# kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# 1. EROSION - Shrinks white regions, removes small white noise
erosion = cv2.erode(binary, kernel, iterations=1)

# 2. DILATION - Expands white regions, fills small holes
dilation = cv2.dilate(binary, kernel, iterations=1)

# 3. OPENING - Erosion followed by Dilation
# Removes small white noise while preserving larger structures
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 4. CLOSING - Dilation followed by Erosion
# Fills small holes while preserving larger structures
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 5. GRADIENT - Difference between dilation and erosion
# Shows the outline/boundary of objects
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# 6. TOP HAT - Difference between input and opening
# Extracts small bright elements
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)

# 7. BLACK HAT - Difference between closing and input
# Extracts small dark elements
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

# Display all results
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary)
cv2.imshow('1. Erosion', erosion)
cv2.imshow('2. Dilation', dilation)
cv2.imshow('3. Opening', opening)
cv2.imshow('4. Closing', closing)
cv2.imshow('5. Gradient', gradient)
cv2.imshow('6. Top Hat', tophat)
cv2.imshow('7. Black Hat', blackhat)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()