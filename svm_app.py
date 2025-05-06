import streamlit as st
import torch
import numpy as np
from PIL import Image
from joblib import load
from torchvision import transforms
from mynet import MyNet  # or from resnet18 import ResNet18

# Configuration
MODEL_PATH = 'hand_model.pth'
SVM_PATH = 'gender_svm.joblib'
IMAGE_SIZE = 64  # Match training size

@st.cache_resource
def load_models():
    """Load both models"""
    # Load classifier
    classifier = MyNet()  # or ResNet18()
    classifier.load_state_dict(torch.load(MODEL_PATH))
    classifier.eval()
    
    # Load SVM
    svm = load(SVM_PATH)
    
    return classifier, svm

def preprocess_image(image):
    """Match training preprocessing"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(page_title="Hand Gender Classifier", layout="centered")
    st.title("✋ Hand Gender Classification")
    
    classifier, svm = load_models()
    
    uploaded_file = st.file_uploader("Upload hand image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        with col2:
            with st.spinner('Analyzing...'):
                try:
                    # Get class probabilities
                    tensor = preprocess_image(image)
                    with torch.no_grad():
                        outputs = classifier(tensor)
                        class_probs = torch.exp(outputs).numpy()[0]
                    
                    # Get gender prediction
                    gender_probs = svm.predict_proba([class_probs])[0]
                    gender = svm.predict([class_probs])[0]
                    
                    # Display results
                    st.subheader("Gender Prediction")
                    st.metric(label="Predicted Gender", 
                            value="Male" if gender == 1 else "Female",
                            delta=f"{max(gender_probs):.1%} confidence")
                    
                    st.progress(gender_probs[0], text=f"Female: {gender_probs[0]:.1%}")
                    st.progress(gender_probs[1], text=f"Male: {gender_probs[1]:.1%}")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    main()