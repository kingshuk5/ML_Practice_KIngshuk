import cv2 
import numpy as np

#load image
image=cv2.imread('Crack.jpg')
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#PreProcessing
blurred= cv2.GaussianBlur(gray,(5,5),0)

#Thresholding
_,thresh=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY_INV)

#Morphological opretions
kernel=np.ones((5,5),np.uint8)
morph=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

#Contour Detection
contours,_=cv2.findContours(morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour)>100:#filter small noise
        x,y,w,h=cv2.boundingRect(contour)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

#save the result
output_path='defect_detected_image.jpg'
cv2.imwrite(output_path,image)

#Display the image
cv2.imshow('Defect Detection',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Detected image saved as:{output_path}")