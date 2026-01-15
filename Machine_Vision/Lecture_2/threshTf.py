import numpy as np
import cv2
img = cv2.imread('dogSep.jpg',0)

ret1, imgT = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
#ret1, imgT = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#If a pixel is > 127 (threshold value) -> set to white else set to black
#Light colors -> Towards 255
#Light colors become white
#dark colors become black

cv2.imshow('Image',img)

cv2.imshow('Thresholded Image',imgT)

cv2.waitKey(0)
cv2.destroyAllWindows()