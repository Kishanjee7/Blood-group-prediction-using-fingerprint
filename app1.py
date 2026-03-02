# ================================
# IMPORTS
# ================================
import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Blood Group Prediction from Fingerprints",
    page_icon="🩸",
    layout="centered"
)

# ================================
# SESSION STATE
# ================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    svm = joblib.load("svm_model.pkl")
    knn = joblib.load("knn_model.pkl")
    meta = joblib.load("meta_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return svm, knn, meta, label_encoder

svm_model, knn_model, meta_model, label_encoder = load_models()

# ================================
# IMAGE PREPROCESSING
# ================================
def enhance_fingerprint(img):
    img = cv2.resize(img, (128,128))
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.createCLAHE(2.0, (8,8)).apply(img)
    img = img / 255.0
    return img

# Convert image to flat features for ML models
def extract_features(img):
    return img.flatten().reshape(1, -1)

# ================================
# HOME PAGE
# ================================
if st.session_state.page == "home":

    st.title("🩸 Blood Group Prediction")
    st.write("Fingerprint-based Blood Group Prediction using Ensemble Machine Learning (SVM + KNN + Meta Model)")

    st.markdown("## Model Architecture")
    st.write("Base Models: SVM + KNN")
    st.write("Final Model: Meta Classifier")

    if st.button("🚀 Start Prediction"):
        st.session_state.page = "predict"
        st.rerun()

# ================================
# PREDICTION PAGE
# ================================
if st.session_state.page == "predict":

    if st.button("⬅ Home"):
        st.session_state.page = "home"
        st.rerun()

    uploaded = st.file_uploader(
        "Upload Fingerprint Image",
        type=["jpg","png","jpeg","bmp"]
    )

    if uploaded:
        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        st.image(img, caption="Original Fingerprint", use_column_width=True)

        enhanced = enhance_fingerprint(img)
        st.image(enhanced, caption="Enhanced Fingerprint", use_column_width=True)

        if st.button("Predict Blood Group"):
            with st.spinner("Analyzing fingerprint..."):

                # Feature extraction
                features = extract_features(enhanced)

                # Base model probabilities
                svm_prob = svm_model.predict_proba(features)
                knn_prob = knn_model.predict_proba(features)

                # Combine base outputs
                final_features = np.concatenate((svm_prob, knn_prob), axis=1)

                # Meta model prediction
                probs = meta_model.predict_proba(final_features)
                pred_class = np.argmax(probs)
                confidence = np.max(probs) * 100
                blood_group = label_encoder.inverse_transform([pred_class])[0]

            st.success(f"🩸 Predicted Blood Group: **{blood_group}**")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

# ================================
# FOOTER
# ================================
st.markdown("""
---
© 2026 Blood Group Prediction from Fingerprints  
Developed by Kishanjee
""")
