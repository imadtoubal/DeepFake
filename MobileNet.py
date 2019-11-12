import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import urllib
import torch.nn as nn

#Load the MobileNet Model
model = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
model.eval()

# File Name of VIdeo
fileName = 'Data/sa1-video-fram1.avi'

#Read Video
cap = cv2.VideoCapture(fileName)

#Get each individual frame from video
success = 1
images = []
while success:
    # vidObj object calls read
    # function extract frames
    success, image = cap.read()
    images.append(image)
image = np.array(images[0])
image = Image.fromarray(image)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

#Removes Output Layer
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

