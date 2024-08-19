import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True) # Load a pre-trained ResNet-18 model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1) # Replace the final layer with a single output for steering angle

    def forward(self, x):
        return self.resnet(x)

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 30

# Define the image transformations for the pre-trained model
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to match the input size of ResNet
    transforms.ToTensor(), # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize using ImageNet statistics
])

# Load the PyTorch model
model = ModifiedResNet()
try:
    model.load_state_dict(torch.load('test_model_second.pth'))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Process incoming telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    print("Telemetry event received.")
    try:
        print(f"Received data: {data}")
        speed = float(data['speed'])
        
        # Decode the image and keep it as a PIL Image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        
        # Debugging: Save and inspect the raw image
        cv2.imwrite('/tmp/raw_image.jpg', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        # Apply the transformation directly on the PIL Image
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Predict the steering angle
        with torch.no_grad():
            image = image.to(next(model.parameters()).device)  # Move image to the same device as model
            steering_angle = float(model(image).item())
            print(f'Raw Steering Angle Output: {steering_angle}')

        throttle = 1.0 - speed / speed_limit

        print(f'Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')
        send_control(steering_angle, throttle)
    except Exception as e:
        print(f"Error in telemetry processing: {e}")


# Handle connection event
@sio.on('connect')
def connect(sid, environ):
    print(f'Client connected: {sid}')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    print(f"Sending control: Steering Angle: {steering_angle}, Throttle: {throttle}")
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    try:
        app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        print("Server started on port 4567")
    except Exception as e:
        print(f"Error starting server: {e}")
