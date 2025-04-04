import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import threading
import time

class EnjoymentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EnjoymentClassifier, self).__init__()

        # TODO 2: construct your own model. CNNs+LSTM is recommended

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
    

        
        self.feature_dim = 128 * 16 * 16 
        self.fc = nn.Linear(self.feature_dim, num_classes)

        
    def forward(self, x):
        

        x = self.conv1(x)  # [B, 32, 64, 64]
        x = self.conv2(x)  # [B, 64, 32, 32]
        x = self.conv3(x)  # [B, 128, 16, 16]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

camera_running = True
current_emotion = 2

def runcameraclassification():
    global camera_running
    global current_emotion
    camera = cv2.VideoCapture(0)
    # Use a pretrained model for the facial detection
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EnjoymentClassifier(3).to(device)
    model.load_state_dict(torch.load("saved_models/model_configs/image_model_50.pth"))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    while camera_running:
        _, frame = camera.read()
        detected_face_in_image = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        face = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(70,70))

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
            cropped_face = frame[max(y-50, 0):y+h+50, max(x-50, 0):x+w+50]
            resized_face = cv2.resize(cropped_face, (300, 300))
            cv2.imshow('Cropped Face', resized_face)
            cropped_face = Image.fromarray(cropped_face)
            
            cropped_face = transform(cropped_face)
            cropped_face = cropped_face.to(device)
            cropped_face = torch.unsqueeze(cropped_face, 0)
            predicted_emotion = model(cropped_face).argmax()
            current_emotion = predicted_emotion
              
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'): # or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1: # This one is for if the window was closed (X button)
            camera_running = False
    
    camera.release()
    cv2.destroyAllWindows()

def printing():
    global camera_running
    global current_emotion
    while camera_running:
        print("Detected Social signal:", ['angry', 'happy', 'neutral'][current_emotion])


def main():
    global camera_running
    t1 = threading.Thread(target=runcameraclassification)
    t2 = threading.Thread(target=printing) #NOTE: Replace this with game.
    t1.start()
    t2.start()
    # Below is experimentation
    time.sleep(10)
    camera_running = False
    t1.join()
    t2.join()
    

if __name__ == '__main__':
    main()