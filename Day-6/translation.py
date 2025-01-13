import cv2
import numpy as np

image = cv2.imread(r'Day-2\images_3.jpg')
rows, cols = image.shape[:2]

# Translation matrix: Shifts by 50 pixels right and 30 pixels down
M = np.float32([[1, 0, 50], [0, 1, 200]])
translated = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow('Translated Image', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
