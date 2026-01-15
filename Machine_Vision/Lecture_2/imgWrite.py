import numpy as np
import cv2
#load an image 
img = cv2.imread('openCVLogo.png',0) # load the image in grayScalled
cv2.imshow('Image', img) # show the converted image

key=cv2.waitKey(0) #wait untill pressing any key 
if key == ord('s'):
    cv2.imwrite('logoGray.png', img) # Save converted image as given name 
cv2.destroyAllWindows()

# Press s for saving the image