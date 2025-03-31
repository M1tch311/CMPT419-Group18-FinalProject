import cv2
import time

# Followed tutorials
# Video capture:
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
# Face detection
# https://www.datacamp.com/tutorial/face-detection-python-opencv

# Open the camera
camera = cv2.VideoCapture(0)

# Use a pretrained model to detect the face within the camera image.
# This allows for easier classification
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

time_taken = 0
frames_computed = 0

while True:
    ret, frame = camera.read()
    detected_face_in_image = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # NOTE: pretty fast. Takes about 0.02-0.03 secs.
    start = time.time_ns()
    # Detect face
    face = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(70,70))
    end = time.time_ns()

    time_taken = frames_computed * time_taken + (end - start)
    frames_computed += 1
    time_taken = time_taken / frames_computed

    # NOTE: For actual running (during game), crop out the face from the image.
    
    biggest_face_box = [0,0,-1,-1]
    # Draw the bounding box around the face in the image
    for (x,y,w,h) in face:
        # Take the biggest image (dont think this is the best solution, )
        if (w + h) > (biggest_face_box[2] + biggest_face_box[3]):
            biggest_face_box = [x,y,w,h]
        cv2.rectangle(detected_face_in_image, (x,y), (x+w, y+h), (0, 255, 0), 4)
    
    # Display
    cv2.imshow('Camera', detected_face_in_image) # Bounding box image
    
    # If we found a face. Show it in a different window
    if biggest_face_box[2] + biggest_face_box[3] > 0:
        cropped_face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (300, 300))
        cv2.imshow('Cropped Face', resized_face)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'): # or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1: # This one is for if the window was closed (X button)
        break

print(time_taken / 1000000000)

# Release the capture and writer objects
camera.release()
cv2.destroyAllWindows()