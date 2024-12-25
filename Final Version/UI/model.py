# PyTorch
import torch

# PyTorch - Others
from torchvision import transforms, models
import torchvision.models as models

from PIL import Image

import cv2
import numpy as np

# Warnings off
import warnings
warnings.filterwarnings("ignore")

# Fix a seed for PyTorch
torch.manual_seed(42)

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate_heatmap(self, class_idx, gamma=0.8):
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = np.maximum(heatmap, 0)  # ReLU to remove negatives
        heatmap /= np.max(heatmap)  # Normalize
        heatmap /= np.max(heatmap)  # Normalize to [0, 1]
        # Apply power transform for better distribution
        heatmap = heatmap**gamma
        return heatmap
        return heatmap

    def visualize(self, image_tensor, class_idx, original_image, save_path=None, colormap=cv2.COLORMAP_JET, gamma=0.8):
        self.model.zero_grad()
        output = self.model(image_tensor)
        class_score = output[0, class_idx]
        class_score.backward()

        heatmap = self.generate_heatmap(class_idx, gamma=gamma)
        heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)


        image_np = np.array(original_image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(image_np, 0.5, heatmap_colored, 0.5, 0)
        output_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, output_overlay)

# Load the model
state_dict = torch.load('best-weighted.pt', map_location=torch.device('cpu'))
model = models.vgg16(num_classes=2)  # Assuming binary classification (Normal, Pneumonia)
model.load_state_dict(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Initialize GradCAM
target_layer = 'features.29'  # This is the last conv layer in VGG16
grad_cam = GradCAM(model, target_layer)

# Preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def test_model(image_path, save_path):

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)

    # Predict the class (Normal or Pneumonia)
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Class labels
    class_labels = {0: "Normal", 1: "Pneumonia"}
    #print(f"Predicted Class: {class_labels[predicted_class]}")

    # Visualize the Grad-CAM heatmap
    prediction = class_labels[predicted_class]
    if prediction == "Pneumonia":
        grad_cam.visualize(input_tensor, predicted_class, original_image, save_path, gamma=0.7)
        
    #return pred_labels[0]
    return prediction





