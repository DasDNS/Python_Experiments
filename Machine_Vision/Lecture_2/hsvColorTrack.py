import cv2
import numpy as np

def nothing(x):
    pass

#Reading the camera

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("Low-H", "Trackbars", 20, 179, nothing)
cv2.createTrackbar("Low-S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Low-V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("High-H", "Trackbars", 40, 179, nothing)
cv2.createTrackbar("High-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("High-V", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_h = cv2.getTrackbarPos("Low-H", "Trackbars")
    low_s = cv2.getTrackbarPos("Low-S", "Trackbars")
    low_v = cv2.getTrackbarPos("Low-V", "Trackbars")
    high_h = cv2.getTrackbarPos("High-H", "Trackbars")
    high_s = cv2.getTrackbarPos("High-S", "Trackbars")
    high_v = cv2.getTrackbarPos("High-V", "Trackbars")

    #Define lower and upper bounds
    lower_ = np.array([low_h, low_s, low_v])
    upper_ = np.array([high_h, high_s, high_v])

    #create a mask
    mask = cv2.inRange(hsv, lower_, upper_)

    #Apply above mask
    result = cv2.bitwise_and(frame, frame, mask=mask)


    cv2.imshow('Original Img', frame)
    cv2.imshow('Mask Img', mask)
    cv2.imshow('Result Img', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    