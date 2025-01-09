import cv2
from PIL import Image
import exifread
from networkx import is_path

#  Capture an image using OpenCV
def capture_image(file_path="captured_photo.jpg"):
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return None

    print("Press 'Spacebar' to capture the photo or 'Esc' to exit.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to exit
            print("Exiting without capturing.")
            break
        elif key == 32:  # Spacebar to capture
            cv2.imwrite(file_path, frame)
            print(f"Photo captured and saved as {file_path}")
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()
    return file_path

image = Image.open(is_path)
exif_data = image._getexif()
if exif_data is not None:
    print("\n Image MetaData;:")
    for tag_id,value in exif_data.items():
        tag= TAGS.get(tag_id,tag_id)
        print(f"{tag}:{value}")
else:
    print("\n No MetaData Found")