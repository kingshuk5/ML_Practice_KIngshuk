import cv2
import numpy as np

image = cv2.imread('Day-2\images_3.jpg')
rows, cols = image.shape[:2]

# Scale image to half size
scaled = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Scaled Image', scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
