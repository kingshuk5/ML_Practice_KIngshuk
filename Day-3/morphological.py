import cv2
import numpy as np
from matplotlib import pyplot as plt

#load the image 
image=cv2.imread('images.jpg',cv2.IMREAD_GRAYSCALE)
original=image.copy()

#apply binary thresholding
_, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

#Define kernel for Morphological Operation
kernel=np.ones((5,5),np.uint8)# 5X Square kernel

#Perform Morphological Operations
erosion= cv2.erode(binary, kernel, iterations=1)
dilation =cv2.dilate(binary, kernel, iterations=1)
opening= cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing= cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient= cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat =cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat= cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

#display Results
plt.figure(figsize=(12,10))

plt.subplot(3,3,1)
plt.title('Original Image')
plt.imshow(original,cmap='gray')

plt.subplot(3,3,2)
plt.title('Binary Image')
plt.imshow(binary,cmap='gray')

plt.subplot(3,3,3)
plt.title('Erosion ')
plt.imshow(erosion,cmap='gray')

plt.subplot(3,3,4)
plt.title('Dilation')
plt.imshow(dilation,cmap='gray')

plt.subplot(3,3,5)
plt.title('Opening ')
plt.imshow(opening,cmap='gray')

plt.subplot(3,3,6)
plt.title('Closing')
plt.imshow(closing,cmap='gray')

plt.subplot(3,3,7)
plt.title('Gradient Image')
plt.imshow(gradient,cmap='gray')

plt.subplot(3,3,8)
plt.title('TopHat')
plt.imshow(tophat,cmap='gray')

plt.subplot(3,3,9)
plt.title('BlackHat')
plt.imshow(blackhat,cmap='gray')

plt.tight_layout()
plt.show()

#save results
cv2.imwrite('eorsion.jpg',erosion)
cv2.imwrite('dilation.jpg',dilation)
cv2.imwrite('opening.jpg',opening)
cv2.imwrite('closing.jpg',closing)
cv2.imwrite('gradient.jpg',gradient)
cv2.imwrite('tophat.jpg',tophat)
cv2.imwrite('blackhat.jpg',blackhat)
