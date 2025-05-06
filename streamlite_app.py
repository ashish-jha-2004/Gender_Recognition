# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from mynet import MyNet
from resnet18 import ResNet18

# Set page config
st.set_page_config(
    page_title="Hand Image Classifier",
    page_icon="✋",
    layout="centered"
)

# Configuration - UPDATE WITH ACTUAL CLASS NAMES
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3',
               'Class 4', 'Class 5', 'Class 6', 'Class 7']

# Custom CSS
st.markdown("""
<style>
    .stApp {max-width: 800px; padding: 2rem;}
    .header {color: #2E86C1; text-align: center; padding: 1rem;}
    .result {font-size: 1.2rem; padding: 1rem; border-radius: 10px; margin-top: 1rem; background-color: #E8F8F5;}
    .confidence-bar {height: 20px; background-color: #85C1E9; margin: 5px 0; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_type='MyNet'):
    """Load the trained model with error handling"""
    try:
        model = MyNet() if model_type == 'MyNet' else ResNet18()
        state_dict = torch.load('hand_model.pth', map_location='cpu')
        
        # Handle potential DataParallel wrapping and architecture mismatches
        state_dict = {k.replace('module.', '').replace('resnet18.', ''): v for k, v in state_dict.items()}
        
        # Load state dict with strict=False to ignore mismatched parameters
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def preprocess_image(image):
    """Consistent preprocessing with training"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Must match original training size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.markdown("<h1 class='header'>✋ Hand Image Classifier</h1>", unsafe_allow_html=True)
    model_type = st.sidebar.selectbox('Model Architecture', ['MyNet', 'ResNet18'])
    
    if (model := load_model(model_type)) is None:
        return

    uploaded_file = st.file_uploader("Upload hand image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, use_column_width=True)
            
            with col2:
                with st.spinner('Analyzing...'):
                    tensor = preprocess_image(image)
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0).numpy()
                        
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                    
                    st.markdown(f"""
                    <div class="result">
                        <div style="font-size: 1.4rem;">
                            Predicted: <strong>{CLASS_NAMES[pred_idx]}</strong>
                        </div>
                        Confidence: <strong>{confidence:.1%}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all class probabilities
                    st.write("**Class Probabilities:**")
                    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
                        st.progress(float(prob), text=f"{cls}: {prob:.1%}")

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()