import cv2
import numpy as np

# Read the image
image = cv2.imread('Lenna.png')

# Check if image was loaded successfully
if image is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# Convert to grayscale (edge detection works on grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Canny Edge Detection (Most popular and effective)
canny_edges = cv2.Canny(gray, 50, 150)

# 2. Sobel Edge Detection (X and Y gradients)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # X direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Y direction
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(sobel_combined)

# 3. Laplacian Edge Detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.abs(laplacian))

# 4. Prewitt Edge Detection (manual implementation)
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittx = cv2.filter2D(gray, -1, kernelx)
prewitty = cv2.filter2D(gray, -1, kernely)
prewitt = prewittx + prewitty

# Display all results
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale', gray)
cv2.imshow('Canny Edges', canny_edges)
cv2.imshow('Sobel Edges', sobel_combined)
cv2.imshow('Laplacian Edges', laplacian)
cv2.imshow('Prewitt Edges', prewitt)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()