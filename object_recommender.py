import os
import csv

import numpy as np
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
    folder = "./datasets/amazon/"+item+"/"
    path = {}
    with open(folder+"data.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            path[row[0]]=row[1]
        
    files = os.listdir(folder)
    files = list(filter(lambda x:x[-3:]=="jpg",files))
    image_features = extract_features(image)
    similarities = []
    for file in files:
        feature = extract_features(os.path.join(folder, file))
        similarity = calculate_similarity(image_features, feature)
        similarities.append(similarity)
    indices = np.array(similarities).argsort()[::-1]
    return [("../"+folder+files[i],path[files[i]]) for i in indices]


# choose_object("lamp","/home/rohan/hackonama/recommendation/TimberlandkingLSWENGE_0d80ca15-a0ad-4341-8b5e-4efa70f4c7a5.webp")