import cv2
import numpy as np 
from  matplotlib import pyplot as plt

#loadthe Image
image = cv2.imread('images.jpg')
original  = image.copy()
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Noise Removal
blurred=cv2.GaussianBlur(gray,(5,5),0)

#Thresholding
_, thresh =cv2.threshold(blurred,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)

#Morphological Operations
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

#sure Background area 
sure_bg=cv2.dilate(opening,kernel,iterations=3)

#sure background area 
dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
_,sure_fg=cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

#unknown region
sure_fg=np.uint8(sure_fg)
unknown=cv2.subtract(sure_bg,sure_fg)

#Marker Labelling

_,markers= cv2.connectedComponents (sure_fg)
markers += 1 #Add 1 to all labels to avoid o as background
markers [unknown == 255] =0 #Mark unknown regions as o

#Visualize Markers Before Watershed
marker_img= cv2.applyColorMap ((markers *10).astype(np.uint8), cv2.COLORMAP_JET)


#Apply Watershed
markers=cv2.watershed(image, markers)
image[markers == -1] =[0, 0, 255] #Mark boundaries in Red

#Display Results at Each Step
plt.figure(figsize=(16, 12))

plt.subplot(2,3,1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))

plt.subplot(2,3,2)
plt.title('Threshold Image')
plt.imshow(thresh,cmap='gray')

plt.subplot(2,3,3)
plt.title('Morphological Image ')
plt.imshow(opening,cmap='gray')

plt.subplot(2,3,4)
plt.title('Sure Foreground')
plt.imshow(sure_fg,cmap='gray')

plt.subplot(2,3,5)
plt.title('Unknown Image')
plt.imshow(unknown,cmap='gray')

plt.subplot(2,3,6)
plt.title('Watershed Segmentation')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

cv2.imwrite('Watershed_segmented.jpg',image)
print("Segmented image saved as 'Watershed_se")
