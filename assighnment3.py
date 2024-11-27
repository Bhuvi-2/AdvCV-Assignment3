# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

#--------------------------------------------------
#    Load Training Data and Testing Data
#--------------------------------------------------

def set_random_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
set_random_seed(0)

#--------------------------------------------------
#       Define Network Architecture
#--------------------------------------------------

# Load CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the projection layer
model.visual.proj = torch.nn.Parameter(model.visual.proj)
model.visual.proj.requires_grad = True

# Define optimizer for the projection layer parameters
optimizer = optim.Adam([model.visual.proj], lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

class_names = [name[13:] for name in glob.glob('./data/train/*')]
class_names = dict(zip(range(len(class_names)), class_names))

# Modifying class names
print("class_names: %s " % class_names)

original_class_names = [
    'Forest', 'Bedroom', 'Office', 'Highway', 'Flower', 
    'Coast', 'InsideCity', 'TallBuilding', 'Industrial', 
    'Street', 'LivingRoom', 'Suburb', 'Mountain', 
    'Kitchen', 'OpenCountry', 'Store'
]

class_mapping = {
    'Forest': 'a forest with trees and greenery',
    'Bedroom': 'a cozy bedroom with furniture',
    'Office': 'a workplace office space',
    'Highway': 'a highway or road',
    'Flower': 'a beautiful flower with petals',
    'Coast': 'a scenic coastline with the ocean',
    'InsideCity': 'a view of a urban building that would be typically be found in a city',
    'TallBuilding': 'a tall skyscraper or building',
    'Industrial': 'an industrial area with factories',
    'Street': 'a street in the city or town',
    'LivingRoom': 'a comfortable living room with chairs or sofas',
    'Suburb': 'a residential house within a suburban neighborhood',
    'Mountain': 'a majestic mountain landscape',
    'Kitchen': 'a kitchen with appliances',
    'OpenCountry': 'an open countryside with a lot of land',
    'Store': 'a shop or store interior'
}

# Generate descriptive labels
descriptive_classes = [class_mapping[cls] for cls in original_class_names]

# Define your augmentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
    #A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
    #A.GaussianBlur(p=0.2),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    #A.Normalize(mean=(0.5,), std=(0.5,))
])

class CustomDataset(Dataset):
    def __init__(self, data, labels, preprocess=None, augment=False):
        self.data = data
        self.labels = labels
        self.preprocess = preprocess
        self.augment = augment

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("L")  # Open as grayscale
        image = image.resize((224, 224), Image.BILINEAR)  # Resize image
        image = image.convert("RGB")  # Convert to RGB by duplicating channels

        if self.augment:
            image = np.array(image)
            image = augmentations(image=image)["image"]
            image = Image.fromarray(image.astype(np.uint8))

        if self.preprocess:
            image = self.preprocess(image)
        else:
            image = transforms.ToTensor()(image)

        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.data)

# Load single training sample per class
def load_single_sample_per_class(path):
    data, labels = [], []
    for id, class_name in class_names.items():
        img_path_class = glob.glob(path + class_name + '/*.jpg')
        if img_path_class:
            data.append(img_path_class[0])  # Append the image path of the first image
            labels.append(id)
    return data, labels

# Load datasets
img_size = (224, 224)
train_data, train_labels = load_single_sample_per_class('./data/train/')
train_dataset = CustomDataset(train_data, train_labels, preprocess=preprocess, augment=True)
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Load test dataset
def load_test_data(path):
    data, labels = [], []
    for id, class_name in class_names.items():
        img_path_class = glob.glob(path + class_name + '/*.jpg')[:50]
        for img_path in img_path_class:
            data.append(img_path)
            labels.append(id)
    return data, labels

test_data, test_labels = load_test_data('./data/test/')
test_dataset = CustomDataset(test_data, test_labels, preprocess=preprocess, augment=False)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#--------------------------------------------------
#       Define Functions for Text Prompts
#--------------------------------------------------
def generate_text_prompts(class_descriptions, templates):
    text_prompts = []
    for desc in class_descriptions:
        prompts = [template.format(desc) for template in templates]
        text_prompts.append(prompts)
    return text_prompts

def compute_text_features(text_prompts, model):
    text_features = []
    with torch.no_grad():
        for prompts in text_prompts:
            tokens = clip.tokenize(prompts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            averaged_embedding = embeddings.mean(dim=0)
            averaged_embedding = averaged_embedding / averaged_embedding.norm()
            text_features.append(averaged_embedding)
    return torch.stack(text_features)

#--------------------------------------------------
#       Training and Evaluation Functions
#--------------------------------------------------

# Training loop to fine-tune the projection layer
def train_projection_layer(model, text_features, trainloader, optimizer, device, epochs=30):
    model.train()

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity and loss
            similarity = image_features @ text_features.T
            loss = nn.CrossEntropyLoss()(similarity, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Update the learning rate after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

# Evaluate model using CLIP's similarity
def evaluate_clip(model, text_features, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Compute image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity and predict labels
            similarity = image_features @ text_features.T
            predictions = similarity.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def reset_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    for param in model.parameters():
        param.requires_grad = False
    model.visual.proj = torch.nn.Parameter(model.visual.proj.clone())
    model.visual.proj.requires_grad = True
    optimizer = optim.Adam([model.visual.proj], lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return model, preprocess, optimizer, scheduler
    
#--------------------------------------------------
#       Run Experiments 
#--------------------------------------------------

# Templates to use
templates_list = [
    ["a photograph of {}"],
    [
        "a photo of {}",
        "an image of {}",
        "a picture of {}",
        "a photograph of {}",
        "an artistic depiction of {}",
        "a close-up photo of {}",
        "a distant view of {}",
        "a rendering of {}",
        "a black and white photo of {}",
        "a low-resolution image of {}",
        "a high-resolution photo of {}",
        "a cropped photo of {}",
    ]
]

# Dictionaries to store results
results = {}

# Experiment 1: Descriptive class tokens with the same prompt
model, preprocess, optimizer, scheduler = reset_model()
print("\nExperiment 1: Training with descriptive class tokens using the same prompt...")
descriptive_class_tokens = torch.cat([clip.tokenize(f"a photograph of {desc}") for desc in descriptive_classes]).to(device)
with torch.no_grad():
    descriptive_class_features = model.encode_text(descriptive_class_tokens)
    descriptive_class_features /= descriptive_class_features.norm(dim=-1, keepdim=True)
train_projection_layer(model, descriptive_class_features, trainloader, optimizer, device)
print("Evaluating with descriptive class features...")
accuracy_exp1 = evaluate_clip(model, descriptive_class_features, testloader, device)
results['Experiment 1'] = accuracy_exp1

# Experiment 2: Original class names with the same prompt
model, preprocess, optimizer, scheduler = reset_model()
print("\nExperiment 2: Training with original class names using the same prompt...")
original_class_tokens = torch.cat([clip.tokenize(f"a photograph of {cls}") for cls in original_class_names]).to(device)
with torch.no_grad():
    original_class_features = model.encode_text(original_class_tokens)
    original_class_features /= original_class_features.norm(dim=-1, keepdim=True)
train_projection_layer(model, original_class_features, trainloader, optimizer, device)
print("Evaluating with original class features...")
accuracy_exp2 = evaluate_clip(model, original_class_features, testloader, device)
results['Experiment 2'] = accuracy_exp2

# Experiment 3: Descriptive class tokens with various text templates
model, preprocess, optimizer, scheduler = reset_model()
print("\nExperiment 3: Training with descriptive class tokens using various text prompts...")
templates = templates_list[1]
text_prompts = generate_text_prompts(descriptive_classes, templates)
descriptive_text_features = compute_text_features(text_prompts, model)
train_projection_layer(model, descriptive_text_features, trainloader, optimizer, device)
print("Evaluating with descriptive class features and various text prompts...")
accuracy_exp3 = evaluate_clip(model, descriptive_text_features, testloader, device)
results['Experiment 3'] = accuracy_exp3

# Experiment 4: Original class names with various text templates
model, preprocess, optimizer, scheduler = reset_model()
print("\nExperiment 4: Training with original class names using various text prompts...")
templates = templates_list[1]
text_prompts = generate_text_prompts(original_class_names, templates)
original_text_features = compute_text_features(text_prompts, model)
train_projection_layer(model, original_text_features, trainloader, optimizer, device)
print("Evaluating with original class features and various text prompts...")
accuracy_exp4 = evaluate_clip(model, original_text_features, testloader, device)
results['Experiment 4'] = accuracy_exp4

# Summary of Results
print("\nSummary of Experiments:")
for exp_name, acc in results.items():
    print(f"{exp_name} Accuracy: {acc:.4f}")