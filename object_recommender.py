import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from scipy.spatial.distance import cosine


model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.eval()

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(model.children())[:-1])  # Exclude the final classification layer

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) 
    
extractor = FeatureExtractor(model)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = extractor(image)
    return features.squeeze().numpy()

def calculate_similarity(room_features, bed_features):
    return 1 - cosine(room_features, bed_features)

def choose_object(item,image):
    folder = "./datasets/amazon/"+item
    files = os.listdir(folder)
    image_features = extract_features(image)
    similarities = []
    for file in files:
        feature = extract_features(os.path.join(folder, file))
        similarity = calculate_similarity(image_features, feature)
        similarities.append(similarity)
    index = similarities.index(max(similarities))
    print(files[index])


choose_object("bed","/home/rohan/hackonama/recommendation/3d-illustration-bedroom-interior-dark-600nw-2258937245.jpg")