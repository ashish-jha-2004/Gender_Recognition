import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from joblib import dump
from torchvision import transforms
from mynet import MyNet  # or from resnet18 import ResNet18

# Configuration - ADJUST THESE TO MATCH YOUR CSV
CSV_PATH = 'HandInfo.csv'
IMAGE_DIR = 'Hands'
MODEL_PATH = 'hand_model.pth'
IMAGE_SIZE = 64  # Match your model's input size
IMAGE_COL = 'imageName'  # Column containing image filenames
GENDER_COL = 'gender'    # Column containing gender labels

def load_model():
    """Load trained classifier model"""
    model = MyNet()  # or ResNet18()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def extract_features(model, image_paths):
    """Extract 8-class probabilities as features"""
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    for img_path in image_paths:
        try:
            image = Image.open(os.path.join(IMAGE_DIR, img_path)).convert('RGB')
            tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.exp(outputs).numpy()[0]
            features.append(probs)
        except Exception as e:
            print(f"Skipping {img_path}: {str(e)}")
    
    return np.array(features)

def main():
    # Load metadata
    df = pd.read_csv(CSV_PATH)
    
    # Clean and map gender values
    df[GENDER_COL] = df[GENDER_COL].str.strip().str.lower()
    gender_map = {'male': 1, 'female': 0}
    df['gender'] = df[GENDER_COL].map(gender_map)
    
    # Load model
    model = load_model()
    
    # Extract features
    X = extract_features(model, df[IMAGE_COL].tolist())
    y = df['gender'].values
    
    # Train SVM
    svm = SVC(probability=True, class_weight='balanced')
    svm.fit(X, y)
    
    # Save model
    dump(svm, 'gender_svm.joblib')
    print(f"SVM trained on {len(X)} samples. Saved as gender_svm.joblib")

if __name__ == '__main__':
    main()