import streamlit as st
import numpy as np
import pickle
from PIL import Image

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

svm_model = load_model()

# -----------------------------
# App Title
# -----------------------------
st.title("🩸 Blood Group Prediction using Fingerprint")

st.write("Upload a fingerprint image to predict blood group.")

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Open image using PIL (no OpenCV required)
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    
    st.image(image, caption="Uploaded Fingerprint", use_column_width=True)

    # -----------------------------
    # Preprocessing (MUST match training)
    # -----------------------------
    
    IMAGE_SIZE = 128  # ⚠️ Change this if your model was trained with different size

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    image_array = np.array(image)
    
    # Normalize
    image_array = image_array / 255.0
    
    # Flatten
    features = image_array.flatten().reshape(1, -1)

    # -----------------------------
    # Feature Size Check
    # -----------------------------
    expected_features = svm_model.n_features_in_

    if features.shape[1] != expected_features:
        st.error(
            f"Feature size mismatch!\n\n"
            f"Model expects {expected_features} features.\n"
            f"But received {features.shape[1]} features.\n\n"
            f"Update IMAGE_SIZE to match training size."
        )
    else:
        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = svm_model.predict(features)[0]
        
        if hasattr(svm_model, "predict_proba"):
            probability = svm_model.predict_proba(features)[0]
            
            st.success(f"Predicted Blood Group: {prediction}")
            st.write("Prediction Probabilities:")
            
            for label, prob in zip(svm_model.classes_, probability):
                st.write(f"{label}: {prob:.4f}")
        else:
            st.success(f"Predicted Blood Group: {prediction}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Developed using Streamlit & Machine Learning")
