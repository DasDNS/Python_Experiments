import cv2

# Read the image
image = cv2.imread('Lenna.png')

# Check if image was loaded successfully
if image is None:
    print("Error: Could not load image. Check the file path.")
else:
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Display the original and grayscale images
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)

    print(type(image))
    print('\n')
    print(image.dtype)

    
    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## Convert to RGB
#rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
# Display the original and grayscale images
#cv2.imshow('Original Image', image)
#cv2.imshow('Grayscale Image', rgb_image)