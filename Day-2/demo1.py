import cv2
import numpy as np
from PIL import Image 
from PIL.ExifTags import TAGS
import matplotlib.pyplot  as plt
import os

#Image_path
image_path =r'C:\Users\kings\OneDrive\Desktop\Industry_training_084\image_5.jpg'


#Check if the image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error : Image not found at : {image_path}")


# Load the RGB image
rgb_image =cv2.imread(image_path)
                      
if rgb_image  is None:
    raise ValueError(f"Error: Unable to load image from path : {image_path}")


# BGR To RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)


#RGB TO GrayScale
gray_image =cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray_image.jpg',gray_image)
# cv2.imshow('Grayscale Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#graytobinary
_, binary_image =cv2.threshold(gray_image,125,255,cv2.THRESH_BINARY)
cv2.imwrite('binary)image.jpg',binary_image)

#RGB image pixel value
height,width,channels =rgb_image.shape
print(f"Image Dimensions:{width}X{height}, Channels:{channels}")
x,y=50,50 #Example 
if x< width and y < height:
    pixel_value  =rgb_image[y,x]
    print(f"Pixel Value at ({x},{y}:{pixel_value})")
else:
    print(f"Coordinates ({x},{y}) are out of bounds")

#image Histogram
plt.figure(figsize=(10,5))
color =('r','g','b')
for i,col in enumerate(color):
    hist = cv2.calcHist([rgb_image],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.title('Histogram from RGB Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()


#pixel Manipulation
for i in range(min(50,height)):
    for j in range(min(50,width)):
        rgb_image[i,j]=[255,0,0]
cv2.imwrite('Manipulated_image.jpg',cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR))

#meta Data
image = Image.open(image_path)
exif_data = image._getexif()
if exif_data is not None:
    print("\n Image MetaData;:")
    for tag_id,value in exif_data.items():
        tag= TAGS.get(tag_id,tag_id)
        print(f"{tag}:{value}")
else:
    print("\n No MetaData Found")