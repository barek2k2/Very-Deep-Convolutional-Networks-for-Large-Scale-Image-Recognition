import torch
model = torch.load('content/Very-Deep-Convolutional-Networks-for-Large-Scale-Image-Recognition/models/VdcnnIR_11' , map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

from PIL import Image
from torchvision import transforms

# Define the same transformations used during training
preprocess = transforms.Compose([
    transforms.Resize(256),             # Resize the image
    transforms.CenterCrop(224),         # Crop to 224x224
    transforms.ToTensor(),              # Convert the image to a tensor
    transforms.Normalize(               # Normalize the image
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load a random image
img = Image.open('cat.jpeg')

# Preprocess the image
img_t = preprocess(img)

# Add batch dimension (1, 3, 224, 224)
img_t = img_t.unsqueeze(0)



# Disable gradient calculation for inference
with torch.no_grad():
    output = model(img_t)

# Convert the output to probabilities using softmax
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class (returns the index of the highest probability)
_, predicted_class = torch.max(output, 1)

print(f'Predicted class: {predicted_class.item()}')
