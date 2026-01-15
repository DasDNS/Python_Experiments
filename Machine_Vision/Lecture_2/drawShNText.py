import numpy as np
import cv2
img = cv2.imread('dogSep.jpg',1)


#Draw a line
cv2.line(img,(10, 10), (400,100), (255,255,255), 3)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()