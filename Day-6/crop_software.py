import cv2
import numpy as np

#Initialize global variables
ref_point = []  
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping,clone

    #If left mouse button is pressed, record the starting (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    #If left mouse button is released, record the ending (x, y) coordinates and indicate that cropping is finished
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        #Draw a rectangle around the region of interest
        cv2.rectangle(clone, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", clone)

#Load the image, clone it, and setup the mouse callback function
image=cv2.imread('Day-6\apple_fruit_powder3.jpg')  

if image is None:
    print("Error: No image found. check the file path")
    exit()

clone = image.copy()

#create a window and set the mouse callback function
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

print("Click and drag for cropping. Press 'r' to reset the cropping region")
print("Press 'c' to crop the image,'s' to save the cropped image and 'q' to quit")


cropped_image = None

while True:
    #Display the image and wait for a keypress
    cv2.imshow("image", image)
    Key = cv2.waitKey(1) & 0xFF

    # If the 'r' key is pressed, reset the cropping region
    if Key == ord('r'):
        clone = image.copy()
        ref_point = []
        cropped_image = None
        print("Cropping region reset.Drag again for cropping")

    # If the 'c' key is pressed, crop the image
    elif Key == ord('c'):
        if len(ref_point) == 2:
            #Crop the region of interest
            x_start,y_start = ref_point[0]
            x_end,y_end = ref_point[1]

            #Ensure the cropping region is valid
            x_start,x_end = min(x_start,x_end),max(x_start,x_end)
            y_start,y_end = min(y_start,y_end),max(y_start,y_end)

            cropped_image = image[y_start:y_end, x_start:x_end]
            cv2.imshow("Cropped Image", cropped_image)
            print(f"Crop region: x={x_start},y={y_start},width={x_end-x_start},height={y_end-y_start}")

        else:
            print("Select the region to crop")

    # If the 's' key is pressed, save the cropped image
    elif Key == ord('s'):
        if cropped_image is not None:
            filename ="cropped_image.jpg"
            cv2.imwrite(filename, cropped_image)
            print(f"Cropped image saved as {filename}")
        else:
            print("No cropped image to save.please crop the image first")
    #If the 'q' key is pressed, quit the program
    elif Key == ord('q'):
        print("Exitting the program")
        break   

cv2.destroyAllWindows()