import cv2
import numpy as np

image = cv2.imread('Day-2\images_3.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
