import cv2
import numpy as np
from matplotlib import pyplot as plt

im_gray = cv2.imread("noisedFlower.jpg",  cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(im_gray, (5,5), 1)
th =  cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
plt.imshow(th,cmap='gray')
cv2.imwrite('noiseremoved.jpg',th)