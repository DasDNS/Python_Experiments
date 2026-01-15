import cv2
import numpy as np

# Read the image
image = cv2.imread('coins.jpg')

if image is None:
    print("Error: Could not load image.")
    exit()

# Create a copy for displaying results
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Apply adaptive thresholding or Otsu's thresholding
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Alternative: Adaptive threshold
# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                cv2.THRESH_BINARY, 11, 2)

# Invert if necessary (coins should be white, background black)
# Check if background is lighter than coins
if np.mean(thresh) > 127:
    thresh = cv2.bitwise_not(thresh)

# Apply morphological operations to clean up
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and circularity
coin_count = 0
min_area = 1000  # Minimum area to be considered a coin
max_area = 50000  # Maximum area

for contour in contours:
    area = cv2.contourArea(contour)
    
    # Filter by area
    if area < min_area or area > max_area:
        continue
    
    # Calculate circularity to ensure it's coin-shaped
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Coins are circular, so circularity should be close to 1
    if circularity > 0.7:  # Adjust threshold as needed
        coin_count += 1
        
        # Draw contour and center point
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
        
        # Calculate center of coin
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output, (cX, cY), 5, (255, 0, 0), -1)
            cv2.putText(output, str(coin_count), (cX - 20, cY - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display coin count on image
cv2.putText(output, f'Total Coins: {coin_count}', (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display all processing steps
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale', gray)
cv2.imshow('Blurred', blurred)
cv2.imshow('Threshold', thresh)
cv2.imshow('Morphological Operations', morph)
cv2.imshow('Coin Detection Result', output)

# Save the result
cv2.imwrite('coins_detected.jpg', output)

print(f'Number of coins detected: {coin_count}')
print('Detection complete! Press any key to close windows.')

cv2.waitKey(0)
cv2.destroyAllWindows()