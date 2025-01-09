import cv2
import numpy as np
import matplotlib.pyplot as plt


#load the image
image=cv2.imread('images.jpg')
if image is None:
    raise FileNotFoundError("Image not found.please check")

#convert BGR to RGB
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#convert to HSV Color space
hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 

#Define  Color Range in HSV
lower_red1= np.array([0,100,100])
upper_red1= np.array([10,255,255])
lower_red2= np.array([160,100,100])
upper_red2= np.array([180,255,255])

#create masks for red color

mask1=cv2.inRange(hsv,lower_red1,upper_red1)
mask2=cv2.inRange(hsv,lower_red2,upper_red2)
mask=cv2.bitwise_or(mask1,mask2)

#apply the mask
segmented_image =cv2.bitwise_and(image_rgb,image_rgb,mask=mask)

#Display original and segmented images

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite('Segmented_image.jpg',cv2.cvtColor(segmented_image,cv2.COLOR_RGB2BGR))
print("Segmented image saved as 'Sefmented_image_4.jpg")
