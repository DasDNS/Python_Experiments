import numpy as np
import cv2
#load an image 
img = cv2.imread('dogSep.jpg',0) # load an image 
cv2.imshow('Image', img) # Display the image inside the "Image" named window
print(img.shape)
print(img[100,100])
cv2.waitKey(0) # Wait until pressing a key 
cv2.destroyAllWindows() # close the all windows

#Press any key in the keyboard, close all the windows
#img = cv2.imread('dogSep.jpg',0) # load an image  => 0 for grey image, 1 for coloured
